#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import copy


class YOLOXMutipleHead(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, extra_head=None, num_extra_head=1):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        if extra_head is None:
            extra_head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        self.extra_heads = [copy.deepcopy(extra_head) for _ in range(num_extra_head)]

        self.up_channels = [nn.Sequential(
            nn.Conv2d(int(256 * self.head.width), int(256 * e.width), kernel_size=1),
            nn.Conv2d(int(512 * self.head.width), int(512 * e.width), kernel_size=1),
            nn.Conv2d(int(1024 * self.head.width), int(1024 * e.width), kernel_size=1),
        ) for e in self.extra_heads]

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )

            extra_losses = []
            for extra_index, extra_head in enumerate(self.extra_heads):
                extra_fpn_outs = list(fpn_outs)
                for i in range(len(extra_fpn_outs)):
                    extra_fpn_outs[i] = self.up_channels[extra_index][i](extra_fpn_outs[i])
                extra_fpn_outs = tuple(extra_fpn_outs)
                extra_loss, extra_iou_loss, extra_conf_loss, extra_cls_loss, extra_l1_loss, extra_num_fg = self.extra_heads[extra_index](
                    extra_fpn_outs, targets, x
                )
                extra_losses.append(extra_loss)

            total_loss = loss

            for extra_loss in extra_losses:
                total_loss += extra_loss

            outputs = {
                "total_loss": total_loss,
                "loss": loss,
                "extra_loss_min": min(extra_losses),
                "num_extra_head": len(extra_losses),
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
