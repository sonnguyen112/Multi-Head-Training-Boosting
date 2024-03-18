import skimage
from yolox.exp import get_exp
import torch
import cv2
import os.path as osp
from tools.demo_track import *

from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess
from PIL import Image

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import torch.multiprocessing as mp
from multiprocessing import Pool
def make_parser_2():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
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

def image_yolox_s(img_path1, img_path2):
    args = make_parser_2().parse_args()
    args.path = img_path1
    args.exp_file = 'exps/example/mot/yolox_nano_mix_det.py'
    args.ckpt = 'pretrained/bytetrack_nano_mot17.pth.tar'
    args.fuse = True
    exp = get_exp(args.exp_file, args.name)
    args.device = "cuda"
    model = exp.get_model().to(args.device)
    model.eval()
    ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    trt_file = None
    decoder = None
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    start = time.time()
    outputs, r1, r2 = predictor.batching_inference(img_path1, img_path2)
    end = time.time()
    print(f"Total Inference Time: {end - start}")
    results = []
    
    if outputs[1] is not None:
        output_results = outputs[1]
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        remain_inds = scores > args.track_thresh
        dets = bboxes[remain_inds] / r2
        print("DET",dets.shape)
    return dets
    # for t in online_targets:
    #     tlwh = t.tlwh
    #     tid = t.track_id
    #     vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
    #     if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
    #         results.append(
    #             [tlwh[0], tlwh[1], tlwh[2], tlwh[3]]
    #         )
    # return results
# print(len(image_yolox_s('/home/khoi/Downloads/420055418_7376426735747116_2750210599166939850_n.jpg')))

def SSIM_algo(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    print(image1.shape)
    image2 = cv2.imread(image2_path)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    print(image2.shape)
    height1, width1= image1_gray.shape
    height2, width2= image2_gray.shape
    if height1 != height2 or width1 != width2:
        new_height = min(height1, height2)
        new_width = min(width1, width2)
        image1_gray = np.resize(image1_gray, (new_height, new_width))
        image2_gray = np.resize(image2_gray, (new_height, new_width))
    score, diff = skimage.metrics.structural_similarity(image1_gray,image2_gray,full=True)
    return score*100
    



def dense_vector_representations(image1_path,image2_path):
    model = SentenceTransformer('clip-ViT-B-32')
    image_names = [image1_path, image2_path]
    encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
    processed_images = util.paraphrase_mining_embeddings(encoded_image)
    duplicates = [image for image in processed_images]
    score, image_id1, image_id2 = duplicates[0]
    return score*100

def get_similarity_errors(image1_path,image2_path):
    image1 = cv2.imread(image1_path)
    print(image1.shape)
    image2 = cv2.imread(image2_path)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    print(image2.shape)
    height1, width1= image1_gray.shape
    height2, width2= image2_gray.shape
    if height1 != height2 or width1 != width2:
        new_height = min(height1, height2)
        new_width = min(width1, width2)
        image1_gray = np.resize(image1_gray, (new_height, new_width))
        image2_gray = np.resize(image2_gray, (new_height, new_width))

    image1_norm = image1_gray / np.sqrt(np.sum(image1_gray**2))
    image2_norm = image2_gray / np.sqrt(np.sum(image2_gray**2))
    return np.sum(image1_norm*image2_norm)
start = time.time()
bbox = image_yolox_s('c.png', "d.png")[0]
end = time.time()
print("bbox",bbox)
print("Time",end - start)
img = cv2.imread('d.png')
#Draw bbox tlwh
intbox = tuple(map(int, (bbox[0], bbox[1], bbox[2], bbox[3])))
cv2.rectangle(img, intbox[0:2], intbox[2:4], (255,0,0), 3)
cv2.imshow('image',img)
cv2.waitKey(0)