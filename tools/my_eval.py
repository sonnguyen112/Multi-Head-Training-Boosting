import os
from demo_track import make_parser, get_exp, Predictor
import time
from loguru import logger
import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker, STrack, matching
from yolox.tracking_utils.timer import Timer
from cython_bbox import bbox_overlaps as bbox_ious
import cv2
import os.path as osp
import numpy as np
import copy
import threading

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
                if avg_similarty < 0.95:
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
        if sim_score > 0.95:
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
                if avg_similarty < 0.95:
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

def thread_util(save_online_targets_img, i, stracks, img_info, outputs_dict, r, sim_list, threshold_to_open_yolo):
    img_h, img_w = img_info["raw_img"].shape[:2]
    crop_h, crop_w = save_online_targets_img[i].shape[:2]
    crop_extend = img_info["raw_img"][max(0, int(stracks[i].tlbr[1]) - crop_h):min(
        img_h, int(stracks[i].tlbr[3]) + crop_h), max(0, int(stracks[i].tlbr[0]) - crop_w):min(img_w, int(stracks[i].tlbr[2]) + crop_w)]
    # cv2.imwrite(f"temp/crop_extend_{i}.jpg", crop_extend)
    x_tl, y_tl = max(0, int(stracks[i].tlbr[0]) - crop_w), max(0, int(stracks[i].tlbr[1]) - crop_h)
    result = cv2.matchTemplate(crop_extend, save_online_targets_img[i], cv2.TM_CCOEFF_NORMED)
    # Get the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # print(max_val)
    # The top-left corner of the matched area
    top_left = max_loc
    # Dimensions of the icon
    h, w, _ = save_online_targets_img[i].shape
    output_img = crop_extend[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
    outputs_dict[stracks[i].track_id] = (np.array([x_tl + top_left[0], y_tl + top_left[1], x_tl + top_left[0] + w, y_tl + top_left[1] + h])*r)
    detect_img = save_online_targets_img[i]
    try:
        # sim_score = SSIM_algo(kalman_img, detect_img)
        # sim_score = calculate_similarity_color_histogram(output_img, detect_img)
        sim_score = max_val
        # sift_sim = SSIM_algo(output_img, detect_img)
    except Exception as e:
        print("SSIM error", e)
        sim_score = 0
    if sim_score > threshold_to_open_yolo:
        save_online_targets_img[i] = output_img
    sim_list.append(sim_score)


def predict_new_detection_template(stracks, img, predictor, timer, save_online_targets_img):
    start_time = time.time()
    threshold_to_open_yolo = 0.95
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
    # outputs = STrack.persist_multi_predict(stracks)
    # # print(outputs)
    # if outputs is None:
    #     return [None], img_info
    # outputs = outputs[:, :4]
    # outputs *= r
    # center = outputs[:, :2].copy()
    # outputs[:, 0] = (center[:, 0] - (outputs[:, 2]*outputs[:, 3]) / 2)
    # outputs[:, 1] = (center[:, 1] - outputs[:, 3] / 2)
    # outputs[:, 2] = (center[:, 0] + (outputs[:, 2]*outputs[:, 3]) / 2)
    # outputs[:, 3] = (center[:, 1] + outputs[:, 3] / 2)
    # Display output
    avg_similarity = 0
    sim_list = []
    outputs = []
    open_yolo = False
    outputs_dict = {}
    threads = []
    for i, output in enumerate(stracks):
        # # crop = img_info["raw_img"][max(0, int(stracks[i].tlbr[1])):max(
        # #     0, int(stracks[i].tlbr[3])), max(0, int(stracks[i].tlbr[0])):max(0, int(stracks[i].tlbr[2]))]
        # crop_h, crop_w = save_online_targets_img[i].shape[:2]
        # crop_extend = img_info["raw_img"][max(0, int(stracks[i].tlbr[1]) - crop_h):max(
        #     0, int(stracks[i].tlbr[3]) + crop_h), max(0, int(stracks[i].tlbr[0]) - crop_w):max(0, int(stracks[i].tlbr[2]) + crop_w)]
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
        # outputs.append(np.array([x_tl + top_left[0], y_tl + top_left[1], x_tl + top_left[0] + w, y_tl + top_left[1] + h])*r)
        # # final_img = img_info["raw_img"][y_tl + top_left[1]:y_tl + top_left[1] + h, x_tl + top_left[0]:x_tl + top_left[0] + w]
        # # cv2.imshow("crop ex", crop_extend)
        # # cv2.imshow("Save", save_online_targets_img[i])
        # # cv2.imshow("detect", output_img)
        # # cv2.imshow("final", final_img)

        # # cv2.waitKey(1000)
        # detect_img = save_online_targets_img[i]
        # # cv2.imshow('kalman img', kalman_img)
        # # cv2.imshow("img", detect_img)
        # # cv2.waitKey(1000)
        # # kalman_img = cv2.cvtColor(kalman_img, cv2.COLOR_BGR2GRAY)
        # # detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        # try:
        #     # sim_score = SSIM_algo(kalman_img, detect_img)
        #     sim_score = calculate_similarity_color_histogram(output_img, detect_img)
        #     # sift_sim = SSIM_algo(output_img, detect_img)
        # except Exception as e:
        #     print("SSIM error", e)
        #     sim_score = 0
        # if sim_score > 0.9:
        #     save_online_targets_img[i] = output_img
        # # print("sift similarity", sift_sim)
        # sim_list.append(sim_score)
        # # cv2.waitKey(0)
        thread = threading.Thread(target=thread_util, args=(save_online_targets_img, i, stracks, img_info, outputs_dict, r, sim_list, threshold_to_open_yolo))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    avg_similarity = np.mean(sim_list)
    for strack in stracks:
        outputs.append(outputs_dict[strack.track_id])
    if avg_similarity < threshold_to_open_yolo:
        open_yolo = True
    # print("Average similarity", avg_similarity)
    # iou_matrix = ious(outputs, [stracks[i].tlbr*r for i in range(len(stracks))])
    outputs = np.concatenate((outputs, np.tile([1, 1, 0], (len(outputs), 1))), axis=1).astype(np.float64)
    outputs = torch.from_numpy(outputs)
    end_time = time.time()
    # print("Predict time", end_time - start_time)
    return [outputs], img_info, None, avg_similarity, sim_list, open_yolo


def eval_bytetrack_with_template_single_video(video, current_time, predictor):
    cap = cv2.VideoCapture(osp.join(PATH_VIDEOS_FOR_EVAL, video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
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
            if frame_id < 10 or frame_id % 10 == 0 or countdown_yolo > 0:
                if countdown_yolo == 0:
                    countdown_yolo = 2
                countdown_yolo -= 1
                outputs, img_info = predictor.inference(frame, timer)
                is_use_yolo = True
            else:
                outputs, img_info, iou_matrix, avg_similarty, sim_list, open_yolo = predict_new_detection_template(
                    online_targets, frame, predictor, timer, save_online_targets_img)
                kalman_output = copy.deepcopy(outputs)
                # print("this is sim_list", sim_list)
                # print("this is avg_similarty", avg_similarty)
                if open_yolo:
                    print("Use YOLO")
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
                # print('this is output', outputs[0])
                # exit(0)
                is_use_yolo = False
            else:
                timer.toc()
                is_use_yolo = False
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

def eval_bytetrack_with_template(current_time, predictor):
    avg_fps = 0
    for video in os.listdir(PATH_VIDEOS_FOR_EVAL):
        fps = eval_bytetrack_with_template_single_video(video, current_time, predictor)
        avg_fps += fps
    print("Average FPS: ", avg_fps / len(os.listdir(PATH_VIDEOS_FOR_EVAL)))

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
    
def eval_bytetrack_with_extra_template_single_video(video, current_time, predictor):
    cap = cv2.VideoCapture(osp.join(PATH_VIDEOS_FOR_EVAL, video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
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
                origin_ratio = img_info["ratio"]
                is_use_yolo = True
            else:
                outputs, img_info = predictor.inference(frame, timer)
                output = outputs[0]
                output = output.cpu().numpy()
                scores = output[:, 4] * output[:, 5]
                bboxes = output[:, :4]  # x1y1x2y2
                remain_inds = scores > 0.1
                dets = (bboxes[remain_inds] / img_info["ratio"]) * origin_ratio
                scores = scores[remain_inds]
                prev_tlbrs = list(map(lambda x: x.tlbr, online_targets))
                prev_tlbrs_id = list(map(lambda x: x.track_id, online_targets))
                cur_tlbrs = list(map(lambda x: x / origin_ratio, dets))
                # for i, tlbr in enumerate(prev_tlbrs):
                #     # Normalize data in range [0-img_info["width"], 0-img_info["height"]]
                #     tlbr[0] = max(0, min(tlbr[0], img_info["width"]))
                #     tlbr[1] = max(0, min(tlbr[1], img_info["height"]))
                #     tlbr[2] = max(0, min(tlbr[2], img_info["width"]))
                #     tlbr[3] = max(0, min(tlbr[3], img_info["height"]))
                # for i, tlbr in enumerate(cur_tlbrs):
                #     tlbr[0] = max(0, min(tlbr[0], img_info["width"]))
                #     tlbr[1] = max(0, min(tlbr[1], img_info["height"]))
                #     tlbr[2] = max(0, min(tlbr[2], img_info["width"]))
                #     tlbr[3] = max(0, min(tlbr[3], img_info["height"]))
                # prev_tlbrs = nms(prev_tlbrs)
                # cur_tlbrs = nms(cur_tlbrs)
                test_img = copy.deepcopy(img_info["raw_img"])
                # for tlbr in prev_tlbrs:
                #     cv2.rectangle(test_img, (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3])), (30, 43, 277), 2)
                # for tlbr in cur_tlbrs:
                #     cv2.rectangle(test_img, (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3])), (24, 234, 237), 3)
                _ious = ious(prev_tlbrs, cur_tlbrs)
                cost_matrix = 1 - _ious
                matches, u_prev, u_cur = matching.linear_assignment(cost_matrix, thresh=args.match_thresh)
                # print(matches, u_prev, u_cur)
                new_detect = []
                # for i,j in matches:
                #     cv2.rectangle(test_img, (int(cur_tlbrs[j][0]), int(cur_tlbrs[j][1])), (int(cur_tlbrs[j][2]), int(cur_tlbrs[j][3])), (255, 255, 255), 3)
                #     new_detect.append(np.concatenate((cur_tlbrs[j], [1])))
                for i in u_prev:
                    cv2.rectangle(test_img, (int(prev_tlbrs[i][0]), int(prev_tlbrs[i][1])), (int(prev_tlbrs[i][2]), int(prev_tlbrs[i][3])), (29, 214, 19), 4)
                    if prev_tlbrs_id[i] not in save_online_targets_img:
                        continue
                    img_h, img_w = img_info["raw_img"].shape[:2]
                    crop_h, crop_w = save_online_targets_img[prev_tlbrs_id[i]].shape[:2]
                    extend_tl = (max(0, min(int(prev_tlbrs[i][0]), img_info["width"]) - crop_w), max(0, min(int(prev_tlbrs[i][1]), img_info["height"]) - crop_h))
                    extend_br = (max(0, min(int(prev_tlbrs[i][2]), img_info["width"]) + crop_w), max(0, min(int(prev_tlbrs[i][3]), img_info["height"]) + crop_h))
                    crop_extend = img_info["raw_img"][extend_tl[1]:extend_br[1], extend_tl[0]:extend_br[0]]
                    x_tl, y_tl = extend_tl
                    result = cv2.matchTemplate(crop_extend, save_online_targets_img[prev_tlbrs_id[i]], cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    if max_val < 0.8:
                        continue
                    top_left = max_loc
                    h, w, _ = save_online_targets_img[prev_tlbrs_id[i]].shape
                    output_img = crop_extend[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
                    new_detect.append((np.array([x_tl + top_left[0], y_tl + top_left[1], x_tl + top_left[0] + w, y_tl + top_left[1] + h, 0.11])))
                    # draw bbox of output img on crop_extend
                    # cv2.rectangle(crop_extend, (top_left[0], top_left[1]), (top_left[0] + w, top_left[1] + h), (255, 255, 255), 3)
                    # cv2.rectangle(img_info["raw_img"], (x_tl + top_left[0], y_tl + top_left[1]), (x_tl + top_left[0] + w, y_tl + top_left[1] + h), (255, 255, 255), 3)
                    # cv2.imshow("crop_extend", crop_extend)
                    # cv2.imshow("output_img", output_img)
                    # cv2.imshow("crop", save_online_targets_img[prev_tlbrs_id[i]])
                    # cv2.imshow("img", img_info["raw_img"])
                    # cv2.waitKey(0)
                for i, _ in enumerate(cur_tlbrs):
                    cv2.rectangle(test_img, (int(cur_tlbrs[i][0]), int(cur_tlbrs[i][1])), (int(cur_tlbrs[i][2]), int(cur_tlbrs[i][3])), (181, 159, 13), 5)
                    new_detect.append(np.concatenate((cur_tlbrs[i],[scores[i]]), axis=0))
                # new_detect = nms(new_detect)
                # print(new_detect)
                # for detect in new_detect:
                #     #red 
                #     color = (0, 0, 255)
                #     if detect[4] < args.track_thresh:
                #         #white
                #         color = (255, 255, 255)
                #     cv2.rectangle(test_img, (int(detect[0]), int(detect[1])), (int(detect[2]), int(detect[3])), color, 3)
                # cv2.imshow("test", test_img)
                # cv2.waitKey(0)
                # print(new_detect)
                outputs = np.stack(new_detect)
                outputs[:, :4] *= origin_ratio
                outputs = [torch.from_numpy(outputs)]
                img_info["ratio"] = origin_ratio
                
            # print(img_info["ratio"])
            # test_out = copy.deepcopy(outputs[0])
            # test_out = test_out.cpu().numpy()
            # scores = test_out[:, 4] * test_out[:, 5]
            # bboxes = test_out[:, :4]  # x1y1x2y2
            # remain_inds = scores > args.track_thresh
            # dets = bboxes[remain_inds] / origin_ratio
            # print(dets)
            # test_img = copy.deepcopy(img_info["raw_img"])
            # for det in dets:
            #     cv2.rectangle(test_img, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 0, 255), 2)
            # print(is_use_yolo)
            det_end = time.perf_counter()
            print("Detection: ", det_end - det_start)
            if outputs[0] is not None:
                # ass_start = time.perf_counter()
                online_targets = tracker.update(
                    outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                if is_use_yolo:
                    save_online_targets_img = {}
                    for t in online_targets:
                        detect_img = img_info["raw_img"][max(0, int(t.tlbr[1])):max(
                            0, int(t.tlbr[3])), max(0, int(t.tlbr[0])):max(0, int(t.tlbr[2]))]
                        save_online_targets_img[t.track_id] = detect_img
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
                is_use_yolo = False
            else:
                timer.toc()
                is_use_yolo = False
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
    fps = 1 / timer.average_time
    timer.clear()
    return fps

def eval_bytetrack_with_extra_template(current_time, predictor):
    avg_fps = 0
    for video in os.listdir(PATH_VIDEOS_FOR_EVAL):
        fps = eval_bytetrack_with_extra_template_single_video(video, current_time, predictor)
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
    elif args.demo == "template":
        eval_bytetrack_with_template(current_time, predictor)
    elif args.demo == "extra_template":
        eval_bytetrack_with_extra_template(current_time, predictor)

if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)