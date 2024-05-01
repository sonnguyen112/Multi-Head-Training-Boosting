#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOXDualHead(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, mini_head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        if mini_head is None:
            mini_head = YOLOXHead(80, 0.5)

        self.backbone = backbone
        self.head = head
        self.mini_head = mini_head
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(int(256 * self.head.width), int(256 * self.mini_head.width), kernel_size=1),
            nn.Conv2d(int(512 * self.head.width), int(512 * self.mini_head.width), kernel_size=1),
            nn.Conv2d(int(1024 * self.head.width), int(1024 * self.mini_head.width), kernel_size=1),
        )

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            reduced_fpn_outs = list(fpn_outs)
            for i in range(len(reduced_fpn_outs)):
                reduced_fpn_outs[i] = self.reduce_channels[i](reduced_fpn_outs[i])
            reduced_fpn_outs = tuple(reduced_fpn_outs)
            mini_loss, mini_iou_loss, mini_conf_loss, mini_cls_loss, mini_l1_loss, mini_num_fg = self.mini_head(
                reduced_fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss + mini_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "mini_loss": mini_loss,
                "mini_iou_loss": mini_iou_loss,
                "mini_l1_loss": mini_l1_loss,
                "mini_conf_loss": mini_conf_loss,
                "mini_cls_loss": mini_cls_loss,
                "mini_num_fg": mini_num_fg,
            }
        else:
            outputs = self.head(fpn_outs)
            # reduced_fpn_outs = list(fpn_outs)
            # for i in range(len(reduced_fpn_outs)):
            #     reduced_fpn_outs[i] = self.reduce_channels[i](reduced_fpn_outs[i])
            # reduced_fpn_outs = tuple(reduced_fpn_outs)
            # outputs = self.mini_head(reduced_fpn_outs)

        return outputs
