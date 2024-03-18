import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracking_utils.timer import Timer
from cython_bbox import bbox_overlaps as bbox_ious
import copy
import skimage
import threading


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-sn", "--smaller_name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "-sf",
        "--smaller_exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-sc", "--smaller_ckpt", default=None, type=str, help="smaller_ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--extra_device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5,
                        help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8,
                        help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(
                    x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
    

    def batching_inference(self, images, timer):
        ratios = []
        imgs = []
        for image in images:
            img, ratio = preproc(image, self.test_size, self.rgb_means, self.std)
            ratios.append(ratio)
            imgs.append(torch.from_numpy(img).float().to(self.device))
        batch = torch.stack(imgs)
        # print(batch.shape)
        if self.fp16:
            batch = batch.half()  # to FP16
        print(self.device)
        start = time.time()
        with torch.no_grad():
            timer.tic()
            outputs = self.model(batch)
            end = time.time()
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, ratios

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(
                outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. /
                timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def calculate_direction(center1, center2):
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # Convert the angle to a range between 0 and 360
    angle = angle % 360
    # Convert the angle to 8 directions(0, 45, 90, 135, 180, 225, 270, 315)
    if angle >= 337.5 or angle < 22.5:
        return 0
    elif angle >= 22.5 and angle < 67.5:
        return 45
    elif angle >= 67.5 and angle < 112.5:
        return 90
    elif angle >= 112.5 and angle < 157.5:
        return 135
    elif angle >= 157.5 and angle < 202.5:
        return 180
    elif angle >= 202.5 and angle < 247.5:
        return 225
    elif angle >= 247.5 and angle < 292.5:
        return 270
    elif angle >= 292.5 and angle < 337.5:
        return 315


def calculate_similarity_color_histogram(imgA, imgB):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    # Compute the color histogram
    histA = cv2.calcHist([grayA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([grayB], [0], None, [256], [0, 256])

    # Compute the similarity
    score = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

    return score


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def SSIM_algo(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    height1, width1 = image1_gray.shape[:2]
    height2, width2 = image2_gray.shape[:2]
    if height1 != height2 or width1 != width2:
        new_height = min(height1, height2)
        new_width = min(width1, width2)
        image1_gray = np.resize(image1_gray, (new_height, new_width))
        image2_gray = np.resize(image2_gray, (new_height, new_width))
    score, diff = skimage.metrics.structural_similarity(image1_gray, image2_gray, full=True)
    return score


def calculate_sift_similarity(imageA, imageB):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(imageA, None)
    kp2, des2 = sift.detectAndCompute(imageB, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # ratio test as per Lowe's paper
    good = []
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            good.append(m)
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    # img3 = cv2.drawMatchesKnn(imageA,kp1,imageB,kp2,matches,None,**draw_params)
    # cv2.imshow('image', img3)
    # cv2.waitKey(500)
    return len(good) / len(kp1)
def find_object_by_smaller_yolo(img, smaller_predictor):
    timer = Timer()
    start = time.time()
    outputs, img_info = smaller_predictor.inference(img, timer)
    end = time.time()
    print(f"Total Inference Time: {end - start}")
    exit(0)

def thread_util(save_online_targets_img, i, stracks, img_info, outputs_dict, r, sim_list, threshold_to_open_yolo, smaller_predictor):
    img_h, img_w = img_info["raw_img"].shape[:2]
    crop_h, crop_w = save_online_targets_img[i].shape[:2]
    crop_extend = img_info["raw_img"][max(0, int(stracks[i].tlbr[1]) - crop_h):min(
        img_h, int(stracks[i].tlbr[3]) + crop_h), max(0, int(stracks[i].tlbr[0]) - crop_w):min(img_w, int(stracks[i].tlbr[2]) + crop_w)]
    # cv2.imwrite(f"temp/crop_extend_{i}.jpg", crop_extend)
    # x_tl, y_tl = max(0, int(stracks[i].tlbr[0]) - crop_w), max(0, int(stracks[i].tlbr[1]) - crop_h)
    # result = cv2.matchTemplate(crop_extend, save_online_targets_img[i], cv2.TM_CCOEFF_NORMED)
    # # Get the location of the best match
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # # print(max_val)
    # # The top-left corner of the matched area
    # top_left = max_loc
    # # Dimensions of the icon
    # h, w, _ = save_online_targets_img[i].shape
    # output_img = crop_extend[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
    output_img = find_object_by_smaller_yolo(crop_extend, smaller_predictor)
    cv2.imwrite(f"temp/output_img_{i}.jpg", output_img)
    exit()
    outputs_dict[stracks[i].track_id] = (np.array(
        [x_tl + top_left[0], y_tl + top_left[1], x_tl + top_left[0] + w, y_tl + top_left[1] + h])*r)
    try:
        # sim_score = SSIM_algo(kalman_img, detect_img)
        # sim_score = calculate_similarity_color_histogram(output_img, detect_img)
        sim_score = max_val
    except Exception as e:
        print("SSIM error", e)
        sim_score = 0
    if sim_score > threshold_to_open_yolo:
        save_online_targets_img[i] = output_img
    sim_list.append(sim_score)

def add_padding(img, target_size):
    # Get the image shape
    h, w = img.shape[:2]

    # Calculate the target size
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left

    # Add black padding
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize the padded image to the target size
    resized_img = cv2.resize(padded_img, target_size)

    return resized_img


def predict_new_detection_tiny(stracks, img, predictor,smaller_predictor, timer, save_online_targets_img):
    threshold_to_open_yolo = 0.9
    timer.tic()
    r = min(predictor.test_size[0] / float(img.shape[0]),
            predictor.test_size[1] / float(img.shape[1]))
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = osp.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    img, ratio = preproc(img, predictor.test_size, predictor.rgb_means, predictor.std)
    img_info["ratio"] = ratio
    avg_similarity = 0
    sim_list = []
    outputs = []
    open_yolo = False
    outputs_dict = {}
    threads = []
    # images = []
    # for i, output in enumerate(stracks):
    #     img_h, img_w = img_info["raw_img"].shape[:2]
    #     _, _, crop_w, crop_h = stracks[i].tlwh
    #     crop_w, crop_h = int(crop_w), int(crop_h)
    #     crop_extend = img_info["raw_img"][max(0, int(stracks[i].tlbr[1]) - crop_h):min(
    #     img_h, int(stracks[i].tlbr[3]) + crop_h), max(0, int(stracks[i].tlbr[0]) - crop_w):min(img_w, int(stracks[i].tlbr[2]) + crop_w)]
    #     # print(crop_extend.shape)
    #     images.append(crop_extend)
    # outputs, ratios = smaller_predictor.batching_inference(images, timer)
    # image_bboxs = copy.deepcopy(images)
    # for i,output in enumerate(outputs):
    #     output = output.cpu().numpy()
    #     scores = output[:, 4] * output[:, 5]
    #     bboxes = output[:, :4]  # x1y1x2y2
    #     remain_inds = scores > args.track_thresh
    #     dets = bboxes[remain_inds] / ratios[i]
    #     print(dets)
    #     for det in dets:
    #         cv2.rectangle(image_bboxs[i], (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 0, 255), 2)
    #     cv2.imshow('image', image_bboxs[i])
    #     cv2.waitKey(0)
    # print(outputs)

    images = []
    max_w, max_h = max(map(lambda x: x.tlwh[2], stracks)), max(map(lambda x: x.tlwh[3], stracks))
    target_size = (int(max_w), int(max_h))
    copy_img = np.ones((height, width, 3), dtype=np.uint8) * 114
    copy_img = copy.deepcopy(img_info["raw_img"])
    rear_size = 100
    copy_img[rear_size: height - rear_size, rear_size: width - rear_size] = 114
    cv2.imwrite(f"temp/copy_img.jpg", copy_img)
    for i, output in enumerate(stracks):
        img_h, img_w = img_info["raw_img"].shape[:2]
        _, _, crop_w, crop_h = stracks[i].tlwh
        crop_w, crop_h = int(crop_w), int(crop_h)
        crop_extend = img_info["raw_img"][max(0, int(stracks[i].tlbr[1]) - crop_h):min(
        img_h, int(stracks[i].tlbr[3]) + crop_h), max(0, int(stracks[i].tlbr[0]) - crop_w):min(img_w, int(stracks[i].tlbr[2]) + crop_w)]
        copy_img[max(0, int(stracks[i].tlbr[1]) - crop_h):min(
        img_h, int(stracks[i].tlbr[3]) + crop_h), max(0, int(stracks[i].tlbr[0]) - crop_w):min(img_w, int(stracks[i].tlbr[2]) + crop_w)] = crop_extend
        cv2.imwrite(f"temp/crop_extend_{i}.jpg", crop_extend)
        # extend_h, extend_w = crop_extend.shape[:2]
        # if extend_h > max_h:
        #     max_h = extend_h
        # if extend_w > max_w:
        #     max_w = extend_w
        images.append(crop_extend)
    # cv2.imshow('image', copy_img)
    # cv2.waitKey(0)
    cv2.imwrite(f"temp/origin.jpg", img_info["raw_img"])
    # for i,image in enumerate(images):
    #     images[i] = add_padding(image, target_size)
    
    # output_size_h = int(np.ceil(np.sqrt(len(images)))) * target_size[1]
    # output_size_w = int(np.ceil(np.sqrt(len(images)))) * target_size[0]
    # output_size = (output_size_w, output_size_h)

    # output_img = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # for idx, img in enumerate(images):
    #     x_offset = (idx % int(np.sqrt(len(images)))) * target_size[0]
    #     y_offset = (idx // int(np.sqrt(len(images)))) * target_size[1]
    #     output_img[y_offset:y_offset+target_size[1], x_offset:x_offset+target_size[0]] = img
    
    # cv2.imshow('image', output_img)
    # # cv2.imshow("img", img_info["raw_img"])
    # cv2.waitKey(0)
    output_img = copy_img
    cv2.imwrite(f"temp/output_img.jpg", output_img)
    start = time.time()
    outputs, img_info = smaller_predictor.inference(output_img, timer)
    end = time.time()
    return outputs, img_info
    print(outputs[0].shape)
    print(f"Total Inference Time: {end - start}")
    output = outputs[0]
    output = output.cpu().numpy()
    scores = output[:, 4] * output[:, 5]
    bboxes = output[:, :4]  # x1y1x2y2
    remain_inds = scores > args.track_thresh
    dets = bboxes[remain_inds] / img_info["ratio"]
    print(dets)
    for det in dets:
        cv2.rectangle(output_img, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 0, 255), 2)
    cv2.imshow('image', output_img)
    cv2.waitKey(0)
    exit(0)

    avg_similarity = np.mean(sim_list)
    for strack in stracks:
        outputs.append(outputs_dict[strack.track_id])
    if avg_similarity < threshold_to_open_yolo:
        open_yolo = True
    outputs = np.concatenate(
        (outputs, np.tile([1, 1, 0], (len(outputs), 1))), axis=1).astype(np.float64)
    outputs = torch.from_numpy(outputs)
    return [outputs], img_info, None, avg_similarity, sim_list, open_yolo


def imageflow_demo_tiny(predictor,smaller_predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    iou_matrix = None
    countdown_yolo = 0
    is_use_yolo = False
    prev_online_targets = []
    save_online_targets_img = []
    online_targets = []
    kalman_output = [[]]
    while True:
        sim_list = None
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            det_start = time.perf_counter()
            prev_online_targets = copy.deepcopy(online_targets)
            if frame_id % 10 == 0 or countdown_yolo > 0:
                if countdown_yolo == 0:
                    countdown_yolo = 2
                countdown_yolo -= 1
                outputs, img_info = predictor.inference(frame, timer)
                is_use_yolo = True
            else:
                # outputs, img_info, iou_matrix, avg_similarty, sim_list, open_yolo = predict_new_detection_tiny(
                #     online_targets, frame, predictor,smaller_predictor, timer, save_online_targets_img)
                # if open_yolo:
                #     print("Use YOLO")
                #     outputs, img_info = predictor.inference(frame, timer)
                #     is_use_yolo = True
                outputs, img_info = predict_new_detection_tiny(
                    online_targets, frame, predictor,smaller_predictor, timer, save_online_targets_img)
            det_end = time.perf_counter()
            print("Detection: ", det_end - det_start)
            if outputs[0] is not None:
                # ass_start = time.perf_counter()
                online_targets = tracker.update(
                    outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                if is_use_yolo:
                    save_online_targets_img = []
                    for t in online_targets:
                        detect_img = img_info["raw_img"][max(0, int(t.tlbr[1])):max(
                            0, int(t.tlbr[3])), max(0, int(t.tlbr[0])):max(0, int(t.tlbr[2]))]
                        save_online_targets_img.append(detect_img)
                # print("this is online_targets", online_targets)
                # ass_end = time.perf_counter()
                # print("Asscociate: ", ass_end - ass_start)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    outputs[0], img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time,
                    iou_matrix=iou_matrix,
                    online_targets=online_targets,
                    sim_list=sim_list,
                    is_use_yolo=is_use_yolo,
                    prev_online_targets=prev_online_targets
                )
                is_use_yolo = False
            else:
                timer.toc()
                is_use_yolo = False
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp,smaller_exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    args.extra_device = torch.device("cuda" if args.extra_device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
        smaller_exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
        smaller_exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
        smaller_exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    smaller_model = smaller_exp.get_model().to(args.extra_device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Smaller Model Summary: {}".format(get_model_info(smaller_model, exp.test_size)))
    model.eval()
    smaller_model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
            smaller_ckpt_file = args.smaller_ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        smaller_ckpt = torch.load(smaller_ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        smaller_model.load_state_dict(smaller_ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)
        smaller_model = fuse_model(smaller_model)

    if args.fp16:
        model = model.half()  # to FP16
        if args.extra_device == torch.device("cuda"):
            smaller_model = smaller_model.half()
        # smaller_model = smaller_model.half()

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    smaller_predictor = Predictor(smaller_model, smaller_exp, trt_file, decoder, args.extra_device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo_tiny(predictor, smaller_predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    smaller_exp = get_exp(args.smaller_exp_file, args.smaller_name)

    main(exp,smaller_exp, args)
