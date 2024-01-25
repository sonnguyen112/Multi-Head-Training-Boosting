import os
from demo_track import make_parser, get_exp, Predictor
import time
from loguru import logger
import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracking_utils.timer import Timer
from cython_bbox import bbox_overlaps as bbox_ious
import cv2
import os.path as osp
import numpy as np
import copy

PATH_VIDEOS_FOR_EVAL = "videos_for_eval"
RESULT_FOLDER = "TrackEval/data/trackers/mot_challenge/MOT17-train/ByteTrack/data"

def calculate_similarity_color_histogram(grayA, grayB):

    # Compute the color histogram
    histA = cv2.calcHist([grayA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([grayB], [0], None, [256], [0, 256])

    # Compute the similarity
    score = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

    return score


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
        return [None], img_info
    outputs = outputs[:, :4]
    outputs *= r
    center = outputs[:, :2].copy()
    outputs[:, 0] = (center[:, 0] - (outputs[:, 2]*outputs[:, 3]) / 2)
    outputs[:, 1] = (center[:, 1] - outputs[:, 3] / 2)
    outputs[:, 2] = (center[:, 0] + (outputs[:, 2]*outputs[:, 3]) / 2)
    outputs[:, 3] = (center[:, 1] + outputs[:, 3] / 2)
    #Display output
    avg_similarity = 0
    sim_list = []
    for i,output in enumerate(outputs):
        strack_bbox = stracks[i].tlbr*r
        # Calulate center of output
        output_center = (output[0] + output[2]) / 2, (output[1] + output[3]) / 2
        strack_center = (strack_bbox[0] + strack_bbox[2]) / 2, (strack_bbox[1] + strack_bbox[3]) / 2
        # Update output bbox
        output[0] = strack_bbox[0] + (output_center[0] - strack_center[0])
        output[1] = strack_bbox[1] + (output_center[1] - strack_center[1])
        output[2] = strack_bbox[2] + (output_center[0] - strack_center[0])
        output[3] = strack_bbox[3] + (output_center[1] - strack_center[1])
        #Crop image
        # print(output[0], output[1], output[2], output[3])
        kalman_img = img_info["raw_img"][max(0,int(output[1] / r)):max(0,int(output[3] / r)), max(0,int(output[0] / r)):max(0,int(output[2] / r))]
        detect_img = img_info["raw_img"][max(0,int(stracks[i].tlbr[1])):max(0, int(stracks[i].tlbr[3])), max(0,int(stracks[i].tlbr[0])):max(0,int(stracks[i].tlbr[2]))]
        # cv2.imshow('kalman img', kalman_img)
        # cv2.imshow("img", detect_img)
        sim_score = calculate_similarity_color_histogram(kalman_img, detect_img)
        # print(sim_score)
        sim_list.append(sim_score)
        avg_similarity += sim_score
        # cv2.waitKey(0)
    avg_similarity /= len(outputs)
    # print("Average similarity", avg_similarity)
    # iou_matrix = ious(outputs, [stracks[i].tlbr*r for i in range(len(stracks))])
    outputs = np.concatenate((outputs, np.tile([1, 1, 0], (len(outputs), 1))), axis=1)
    outputs = torch.from_numpy(outputs)
    # print(outputs.shape[1])
    return [outputs], img_info, avg_similarity, sim_list

def eval_bytetrack_with_original_single_video(video, current_time, predictor):
    print(video)
    cap = cv2.VideoCapture(osp.join(PATH_VIDEOS_FOR_EVAL, video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
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
                        results.append(
                            f"{frame_id + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
            else:
                timer.toc()
                online_im = img_info['raw_img']
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(RESULT_FOLDER, f"{video.split('.')[0]}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    return 1. / timer.average_time

def eval_bytetrack_with_kalman_single_video(video, current_time, predictor):
    print(video)
    cap = cv2.VideoCapture(osp.join(PATH_VIDEOS_FOR_EVAL, video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
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
            if frame_id > 10 and (frame_id + 1) % 3 == 0:
                outputs, img_info, avg_similarty, sim_list = predict_new_detection(
                    online_targets, frame, predictor, timer)
                if any(i < 0.9 for i in sim_list):
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
                        results.append(
                            f"{frame_id + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
            else:
                timer.toc()
                online_im = img_info['raw_img']
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(RESULT_FOLDER, f"{video.split('.')[0]}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    return 1. / timer.average_time


def eval_bytetrack_with_kalman(current_time, predictor):
    avg_fps = 0
    for video in os.listdir(PATH_VIDEOS_FOR_EVAL):
        fps = eval_bytetrack_with_kalman_single_video(video, current_time, predictor)
        avg_fps += fps
    print("Average FPS: ", avg_fps / len(os.listdir(PATH_VIDEOS_FOR_EVAL)))

def eval_bytetrack_with_origin(current_time, predictor):
    avg_fps = 0
    for video in os.listdir(PATH_VIDEOS_FOR_EVAL):
        fps = eval_bytetrack_with_original_single_video(video, current_time, predictor)
        avg_fps += fps
    print("Average FPS: ", avg_fps / len(os.listdir(PATH_VIDEOS_FOR_EVAL)))

def predict_new_detection_2(stracks, img, predictor, timer, save_online_targets_img):
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
            output[1] / r)):max(0, int(output[3] / r)), max(0, int(output[0] / r)):max(0, int(output[2] / r))]
        # detect_img = img_info["raw_img"][max(0, int(stracks[i].tlbr[1])):max(
        #     0, int(stracks[i].tlbr[3])), max(0, int(stracks[i].tlbr[0])):max(0, int(stracks[i].tlbr[2]))]
        detect_img = save_online_targets_img[i]
        # cv2.imshow('kalman img', kalman_img)
        # cv2.imshow("img", detect_img)
        # cv2.waitKey(1000)
        # kalman_img = cv2.cvtColor(kalman_img, cv2.COLOR_BGR2GRAY)
        # detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        try:
            # sim_score = SSIM_algo(kalman_img, detect_img)
            sim_score = calculate_similarity_color_histogram(kalman_img, detect_img)
        except:
            sim_score = 0
        if sim_score > 0.9:
            save_online_targets_img[i] = kalman_img
        # print(sim_score)
        sim_list.append(sim_score)
        avg_similarity += sim_score
        # cv2.waitKey(0)
    avg_similarity /= len(outputs)
    # # print("Average similarity", avg_similarity)
    # iou_matrix = ious(outputs, [stracks[i].tlbr*r for i in range(len(stracks))])
    outputs = np.concatenate((outputs, np.tile([1, 1, 0], (len(outputs), 1))), axis=1)
    outputs = torch.from_numpy(outputs)
    # print(outputs.shape[1])
    return [outputs], img_info, None, avg_similarity, sim_list

def eval_bytetrack_with_kalman_2_single_video(video, current_time, predictor):
    print(video)
    cap = cv2.VideoCapture(osp.join(PATH_VIDEOS_FOR_EVAL, video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    countdown_yolo = 0
    is_use_yolo = False
    prev_online_targets = []
    save_online_targets_img = []
    online_targets = []
    kalman_output = [[]]
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            det_start = time.perf_counter()
            if frame_id < 10 or frame_id % 10 == 0 or countdown_yolo > 0:
                if countdown_yolo == 0:
                    countdown_yolo = 2
                countdown_yolo -= 1
                outputs, img_info = predictor.inference(frame, timer)
                is_use_yolo = True
            else:
                outputs, img_info, iou_matrix, avg_similarty, sim_list = predict_new_detection_2(
                    online_targets, frame, predictor, timer, save_online_targets_img)
                kalman_output = copy.deepcopy(outputs)
                # print("this is sim_list", sim_list)
                # print("this is avg_similarty", avg_similarty)
                if avg_similarty < 0.9 or any(i < 0.9 for i in sim_list):
                    outputs, img_info = predictor.inference(frame, timer)
                    is_use_yolo = True
            # if frame_id < 20 or frame_id % 5 == 0:
            #     outputs, img_info = predictor.inference(frame, timer)
            # else:
            #     outputs, img_info, iou_matrix, avg_similarty = predict_new_detection(
            #         online_targets, frame, pcolorredictor, timer)
            #     if avg_similarty < 0.9:
            #         outputs, img_info = predictor.inference(frame, timer)
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
            else:
                timer.toc()
                online_im = img_info['raw_img']
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(RESULT_FOLDER, f"{video.split('.')[0]}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    return 1. / timer.average_time

def eval_bytetrack_with_kalman_2(current_time, predictor):
    avg_fps = 0
    for video in os.listdir(PATH_VIDEOS_FOR_EVAL):
        fps = eval_bytetrack_with_kalman_2_single_video(video, current_time, predictor)
        avg_fps += fps
    print("Average FPS: ", avg_fps / len(os.listdir(PATH_VIDEOS_FOR_EVAL)))

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

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

    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "kalman":
        eval_bytetrack_with_kalman(current_time, predictor)
    elif args.demo == "origin":
        eval_bytetrack_with_origin(current_time, predictor)
    elif args.demo == "kalman_2":
        eval_bytetrack_with_kalman_2(current_time, predictor)

if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)