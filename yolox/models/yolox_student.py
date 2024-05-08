#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import torch
from .GloRe import GloRe_Unit_2D

class YOLOXStudent(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

        self.cls_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
        ])

        self.reg_adaptation = nn.ModuleList([
            nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0),
        ])

        self.obj_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
        ])


    def forward(self, x, targets=None, t_model = None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), cls_outputs, reg_outputs, obj_outputs = self.head.kf_forward(
                fpn_outs, targets, x
            )        

            with torch.no_grad():
                t_fpn_outs = t_model.backbone(x)
                _, t_cls_outputs, t_reg_outputs, t_obj_outputs = t_model.head.kf_forward(t_fpn_outs)
            
            # print(cls_outputs[0].shape, t_cls_outputs[0].shape)
            # print(reg_outputs[0].shape, t_reg_outputs[0].shape)
            # print(obj_outputs[0].shape, t_obj_outputs[0].shape)

            cls_kd_loss = 0
            reg_kd_loss = 0
            obj_kd_loss = 0

            for i in range(3):
               cls_kd_loss += torch.dist(self.cls_adaptation[i](cls_outputs[i]), t_cls_outputs[i], 2)
               reg_kd_loss += torch.dist(self.reg_adaptation[i](reg_outputs[i]), t_reg_outputs[i], 2)
               obj_kd_loss += torch.dist(self.obj_adaptation[i](obj_outputs[i]), t_obj_outputs[i], 2)

            outputs = {
                "total_loss": loss + cls_kd_loss + reg_kd_loss + obj_kd_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "cls_kd_loss": cls_kd_loss,
                "reg_kd_loss": reg_kd_loss,
                "obj_kd_loss": obj_kd_loss
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
