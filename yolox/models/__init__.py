#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
from .yolox_student import YOLOXStudent
from .yolox_student_1 import YOLOXStudent1
from .yolox_dual_head import YOLOXDualHead
from .yolox_triple_head import YOLOXTripleHead
from .yolox_five_head import YOLOXFiveHead
from .yolox_triple_head_advance import YOLOXTripleHeadAdvance
from .yolox_n_head import YOLOXMutipleHead