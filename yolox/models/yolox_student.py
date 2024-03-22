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

        self.student_non_local = nn.ModuleList(
            [   
                GloRe_Unit_2D(128,256),
                GloRe_Unit_2D(256,256),
                GloRe_Unit_2D(512,256),
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                GloRe_Unit_2D(128,256),
                GloRe_Unit_2D(256,256),
                GloRe_Unit_2D(512,256),
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        ])
        self.for_adaptation = nn.ModuleList([
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        ])

    def forward(self, x, targets=None, t_model = None):
        # fpn output content features of [dark3, dark4, dark5]
        # print(t_model)
        fpn_outs = self.backbone(x)
        print(fpn_outs[0].shape, fpn_outs[1].shape, fpn_outs[2].shape)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )

            kd_nonlocal_loss = 0
            kd_foreground_loss=0

            for i in range(3):
                student_feature = fpn_outs[i]
                teacher_feature = t_model.backbone(x)[i]
                print(teacher_feature[0].shape, teacher_feature[1].shape, teacher_feature[2].shape)
                s_relation = self.student_non_local[i](student_feature)
                t_relation = self.teacher_non_local[i](teacher_feature)
                # print(s_relation.shape, t_relation.shape)
                # print(student_feature.shape, teacher_feature.shape)
                kd_nonlocal_loss += torch.dist(self.non_local_adaptation[i](s_relation), t_relation, p=2)
                # kd_foreground_loss += torch.dist(student_feature, teacher_feature, p=2)
                kd_foreground_loss += torch.dist(self.for_adaptation[i](student_feature), teacher_feature, p=2)
            kd_nonlocal_loss *= 0.004
            kd_foreground_loss *= 0.006
                
            outputs = {
                "total_loss": loss + kd_foreground_loss + kd_nonlocal_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "kd_foreground_loss": kd_foreground_loss,
                "kd_nonlocal_loss": kd_nonlocal_loss
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
