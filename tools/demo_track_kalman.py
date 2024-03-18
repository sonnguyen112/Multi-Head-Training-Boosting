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
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
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


def predict_new_detection(stracks, img, predictor, timer):
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
    outputs = STrack.persist_multi_predict(stracks)
    # print(outputs)
    if outputs is None:
        return [None], img_info, None, 0, [0]
    outputs = outputs[:, :4]
    outputs *= r
    center = outputs[:, :2].copy()
    outputs[:, 0] = (center[:, 0] - (outputs[:, 2]*outputs[:, 3]) / 2)
    outputs[:, 1] = (center[:, 1] - outputs[:, 3] / 2)
    outputs[:, 2] = (center[:, 0] + (outputs[:, 2]*outputs[:, 3]) / 2)
    outputs[:, 3] = (center[:, 1] + outputs[:, 3] / 2)
    # Display output
    avg_similarity = 0
    sim_list = []
    for i, output in enumerate(outputs):
        strack_bbox = stracks[i].tlbr*r
        # Calulate center of output
        output_center = (output[0] + output[2]) / 2, (output[1] + output[3]) / 2
        strack_center = (strack_bbox[0] + strack_bbox[2]) / 2, (strack_bbox[1] + strack_bbox[3]) / 2
        # Update output bbox
        output[0] = strack_bbox[0] + (output_center[0] - strack_center[0])
        output[1] = strack_bbox[1] + (output_center[1] - strack_center[1])
        output[2] = strack_bbox[2] + (output_center[0] - strack_center[0])
        output[3] = strack_bbox[3] + (output_center[1] - strack_center[1])
        # Crop image
        # print(output[0], output[1], output[2], output[3])
        kalman_img = img_info["raw_img"][max(0, int(
            output[1] / r)):min(height, int(output[3] / r)), max(0, int(output[0] / r)):min(width, int(output[2] / r))]
        detect_img = img_info["raw_img"][max(0, int(stracks[i].tlbr[1])):min(
            height, int(stracks[i].tlbr[3])), max(0, int(stracks[i].tlbr[0])):min(width, int(stracks[i].tlbr[2]))]
        # cv2.imshow('kalman img', kalman_img)
        # cv2.imshow("img", detect_img)
        sim_score = calculate_similarity_color_histogram(kalman_img, detect_img)
        # print(sim_score)
        sim_list.append(sim_score)
        avg_similarity += sim_score
        # cv2.waitKey(0)
    avg_similarity /= len(outputs)
    # print("Average similarity", avg_similarity)
    iou_matrix = ious(outputs, [stracks[i].tlbr*r for i in range(len(stracks))])
    outputs = np.concatenate((outputs, np.tile([1, 1, 0], (len(outputs), 1))), axis=1)
    outputs = torch.from_numpy(outputs)
    # print(outputs.shape[1])
    return [outputs], img_info, iou_matrix, avg_similarity, sim_list

def imageflow_demo(predictor, vis_folder, current_time, args):
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
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            det_start = time.perf_counter()
            outputs, img_info = predictor.inference(frame, timer)
            det_end = time.perf_counter()
            print("Detection: ", det_end - det_start)
            if outputs[0] is not None:
                # ass_start = time.perf_counter()
                online_targets = tracker.update(
                    outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                # print(online_targets)
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
                # print('this is output', outputs[0])
                # exit(0)
                online_im = plot_tracking(
                    outputs[0], img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
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


def imageflow_demo_kalman(predictor, vis_folder, current_time, args):
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
    while True:
        sim_list = None
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            det_start = time.perf_counter()
            if frame_id > 10 and (frame_id + 1) % 3 == 0:
                outputs, img_info, iou_matrix, avg_similarty, sim_list = predict_new_detection(
                    online_targets, frame, predictor, timer)
                if avg_similarty < 0.9:
                    outputs, img_info = predictor.inference(frame, timer)
            else:
                outputs, img_info = predictor.inference(frame, timer)
            # if frame_id < 20 or frame_id % 5 == 0:
            #     outputs, img_info = predictor.inference(frame, timer)
            # else:
            #     outputs, img_info, iou_matrix, avg_similarty = predict_new_detection(
            #         online_targets, frame, predictor, timer)
            #     if avg_similarty < 0.9:
            #         outputs, img_info = predictor.inference(frame, timer)
            det_end = time.perf_counter()
            print("Detection: ", det_end - det_start)
            if outputs[0] is not None:
                # ass_start = time.perf_counter()
                online_targets = tracker.update(
                    outputs[0], [img_info['height'], img_info['width']], exp.test_size)
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
                # print('this is output', outputs[0])
                # exit(0)
                online_im = plot_tracking(
                    outputs[0], img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time,
                    iou_matrix=iou_matrix,
                    online_targets=online_targets,
                    sim_list=sim_list
                )
            else:
                timer.toc()
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
    matches = bf.knnMatch(des1,des2,k=2)
    # ratio test as per Lowe's paper
    good = []
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            good.append(m)
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
    # img3 = cv2.drawMatchesKnn(imageA,kp1,imageB,kp2,matches,None,**draw_params)
    # cv2.imshow('image', img3)
    # cv2.waitKey(500)
    return len(good) / len(kp1)


def main(exp, args):
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

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

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
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo_kalman(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
