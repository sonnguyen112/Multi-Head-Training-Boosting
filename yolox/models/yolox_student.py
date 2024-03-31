#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import torch
from .GloRe import GloRe_Unit_2D
import math


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
                GloRe_Unit_2D(128,128),
                GloRe_Unit_2D(256,256),
                GloRe_Unit_2D(512,512),
            ]
        )
        self.student_non_local_reg_head = nn.ModuleList(
            [   
                GloRe_Unit_2D(128,128),
                GloRe_Unit_2D(128,128),
                GloRe_Unit_2D(128,128),
            ]
        )
        self.student_non_local_cls_head = nn.ModuleList(
            [   
                GloRe_Unit_2D(128,128),
                GloRe_Unit_2D(128,128),
                GloRe_Unit_2D(128,128),
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                GloRe_Unit_2D(320,320),
                GloRe_Unit_2D(640,640),
                GloRe_Unit_2D(1280,1280),
            ]
        )
        self.teacher_non_local_reg_head = nn.ModuleList(
            [
                GloRe_Unit_2D(320,320),
                GloRe_Unit_2D(320,320),
                GloRe_Unit_2D(320,320),
            ]
        )
        self.teacher_non_local_cls_head = nn.ModuleList(
            [
                GloRe_Unit_2D(320,320),
                GloRe_Unit_2D(320,320),
                GloRe_Unit_2D(320,320),
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1280, kernel_size=1, stride=1, padding=0),
        ])
        self.non_local_adaptation_reg_head = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
        ])
        self.non_local_adaptation_cls_head = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
        ])
        self.for_adaptation = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1280, kernel_size=1, stride=1, padding=0),
        ])
        self.reg_head_adaptation = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
        ])
        self.cls_head_adaptation = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
        ])
        self.out_head_adaptation = nn.ModuleList([
            nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0),
        ])


    def forward(self, x, targets=None, t_model = None):
        # fpn output content features of [dark3, dark4, dark5]
        # print(t_model)
        fpn_outs = self.backbone(x)
        # if t_model is not None:
        #     test = t_model.head.raw_inference(fpn_outs)
        #     print(fpn_outs[0].shape)
        #     print(test[0].shape)
        #     exit()
        # print(fpn_outs[0].shape, fpn_outs[1].shape, fpn_outs[2].shape)
        if self.training:
            assert targets is not None
            (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), s_reg_head_feat, s_cls_head_feat, s_output_head = self.head.kf_forward(
                fpn_outs, targets, x
            )

            kd_nonlocal_loss = 0
            kd_foreground_loss = 0
            kd_reg_head_loss = 0
            kd_glore_reg_loss = 0
            kd_cls_head_loss = 0
            kd_glore_cls_loss = 0
            kd_out_head_loss = 0
            t_feat = t_model.backbone(x)
            t_reg_head_feat, t_cls_head_feat, t_output_head = t_model.head.kf_forward(t_feat)

            for i in range(3):
                student_feature = fpn_outs[i]
                teacher_feature = t_feat[i]
                # Normalize the feature by min-max
                student_feature = (student_feature - student_feature.min()) / (student_feature.max() - student_feature.min())
                teacher_feature = (teacher_feature - teacher_feature.min()) / (teacher_feature.max() - teacher_feature.min())
                s_relation = self.student_non_local[i](student_feature)
                t_relation = self.teacher_non_local[i](teacher_feature)
                # print(s_relation.shape, t_relation.shape)
                # print(student_feature.shape, teacher_feature.shape)
                kd_nonlocal_loss += torch.dist(self.non_local_adaptation[i](s_relation), t_relation, p=2)
                kd_foreground_loss += torch.dist(self.for_adaptation[i](student_feature), teacher_feature, p=2)

                student_reg_feat = s_reg_head_feat[i]
                teacher_reg_feat = t_reg_head_feat[i]
                student_cls_feat = s_cls_head_feat[i]
                teacher_cls_feat = t_cls_head_feat[i]
                student_out = s_output_head[i]
                teacher_out = t_output_head[i]
                # Normalize the feature by min-max
                student_reg_feat = (student_reg_feat - student_reg_feat.min()) / (student_reg_feat.max() - student_reg_feat.min())
                teacher_reg_feat = (teacher_reg_feat - teacher_reg_feat.min()) / (teacher_reg_feat.max() - teacher_reg_feat.min())
                student_cls_feat = (student_cls_feat - student_cls_feat.min()) / (student_cls_feat.max() - student_cls_feat.min())
                teacher_cls_feat = (teacher_cls_feat - teacher_cls_feat.min()) / (teacher_cls_feat.max() - teacher_cls_feat.min())
                student_out = (student_out - student_out.min()) / (student_out.max() - student_out.min())
                teacher_out = (teacher_out - teacher_out.min()) / (teacher_out.max() - teacher_out.min())

                kd_reg_head_loss += torch.dist(self.reg_head_adaptation[i](student_reg_feat), teacher_reg_feat, p=2)
                kd_cls_head_loss += torch.dist(self.cls_head_adaptation[i](student_cls_feat), teacher_cls_feat, p=2)
                kd_out_head_loss += torch.dist(self.out_head_adaptation[i](student_out), teacher_out, p=2)

                s_reg_head_relation = self.student_non_local_reg_head[i](student_reg_feat)
                t_reg_head_relation = self.teacher_non_local_reg_head[i](teacher_reg_feat)
                kd_glore_reg_loss += torch.dist(self.non_local_adaptation_reg_head[i](s_reg_head_relation), t_reg_head_relation, p=2)

                s_cls_head_relation = self.student_non_local_cls_head[i](student_cls_feat)
                t_cls_head_relation = self.teacher_non_local_cls_head[i](teacher_cls_feat)
                kd_glore_cls_loss += torch.dist(self.non_local_adaptation_cls_head[i](s_cls_head_relation), t_cls_head_relation, p=2)


            kd_nonlocal_loss *= 0.004
            kd_foreground_loss *= 0.006
            kd_reg_head_loss *= 0.005
            kd_glore_reg_loss *= 0.005
            kd_cls_head_loss *= 0.005
            kd_glore_cls_loss *= 0.005
            kd_out_head_loss *= 0.005
                
            outputs = {
                "total_loss": loss + kd_foreground_loss + kd_nonlocal_loss + kd_reg_head_loss + kd_glore_reg_loss + kd_cls_head_loss + kd_glore_cls_loss + kd_out_head_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "kd_foreground_loss": kd_foreground_loss,
                "kd_nonlocal_loss": kd_nonlocal_loss,
                "kd_reg_head_loss": kd_reg_head_loss,
                "kd_glore_reg_loss": kd_glore_reg_loss,
                "kd_cls_head_loss": kd_cls_head_loss,
                "kd_glore_cls_loss": kd_glore_cls_loss,
                "kd_out_head_loss": kd_out_head_loss
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
