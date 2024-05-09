#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import torch


class YOLOXDualHead(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, extra_backbone=None, extra_head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        self.extra_backbone = extra_backbone
        self.extra_head = extra_head
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(int(256 * self.head.width), 1, kernel_size=1),
            nn.Conv2d(int(512 * self.head.width), 1, kernel_size=1),
            nn.Conv2d(int(1024 * self.head.width), 1, kernel_size=1),
        )
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False),
        )

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )

            fpn_outs_1 = list(fpn_outs)
            for i in range(len(fpn_outs_1)):
                fpn_outs_1[i] = self.up_sample[i](fpn_outs_1[i])
                fpn_outs_1[i] = self.reduce_channels[i](fpn_outs_1[i])
            extra_input = torch.cat(fpn_outs_1, dim=1)
            extra_fpn_outs = self.extra_backbone(extra_input)
            extra_loss, extra_iou_loss, extra_conf_loss, extra_cls_loss, extra_l1_loss, extra_num_fg = self.extra_head(
                extra_fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss + extra_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "extra_loss": extra_loss,
                "extra_iou_loss": extra_iou_loss,
                "extra_l1_loss": extra_l1_loss,
                "extra_conf_loss": extra_conf_loss,
                "extra_cls_loss": extra_cls_loss,
                "extra_num_fg": extra_num_fg,
            }
        else:
            outputs = self.head(fpn_outs)
            # reduced_fpn_outs = list(fpn_outs)
            # for i in range(len(reduced_fpn_outs)):
            #     reduced_fpn_outs[i] = self.reduce_channels[i](reduced_fpn_outs[i])
            # reduced_fpn_outs = tuple(reduced_fpn_outs)
            # outputs = self.mini_head(reduced_fpn_outs)

        return outputs
