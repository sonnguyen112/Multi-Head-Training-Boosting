from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import TeacherStudentTrainer, launch
from yolox.exp import get_exp

import argparse
import random
import warnings

from thop import profile

from copy import deepcopy

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "-tf",
        "--teacher_exp_file",
        default=None,
        type=str,
        help="plz input your teacher expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-tc", "--teacher_ckpt", default=None, type=str, help="teacher checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def get_model_info(model, tsize):

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info

def init_teacher_model(self,t_model):
    logger.info("Init teacher model")
    ckpt_file = self.args.teacher_ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    t_model.load_state_dict(ckpt["model"])
    logger.info("Create Teacher done")
    return t_model

if __name__ == "__main__":
    args = make_parser().parse_args()
    t_exp = get_exp(args.teacher_exp_file, None)

    if t_exp.seed is not None:
        random.seed(t_exp.seed)
        torch.manual_seed(t_exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )
    t_model = t_exp.get_model()
    logger.info(
            "Teacher Model Summary: {}".format(get_model_info(t_model, t_exp.test_size))
        )
    t_model.to("cuda:0")
    t_model = init_teacher_model(t_model)
    no_aug = start_epoch >= max_epoch - t_exp.no_aug_epochs
    train_loader = t_exp.get_data_loader(
            batch_size=args.batch_size,
            is_distributed=False,
            no_aug=no_aug,
        )
    logger.info("init prefetcher, this might take one minute or less...")
    self.prefetcher = DataPrefetcher(self.train_loader)