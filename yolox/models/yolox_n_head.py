#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .GloRe import GloRe_Unit_2D
import copy
import torch
from torchvision.ops import roi_align


class ObjectRelation_1(nn.Module):
    def __init__(self, in_channels):
        super(ObjectRelation_1, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels//2
        self.softmax = nn.Softmax(-1)
        self.theta = nn.Sequential(
            nn.Linear(self.in_channels, self.inter_channels))
        self.phi = nn.Sequential(
            nn.Linear(self.in_channels, self.inter_channels))
        self.g = nn.Sequential(
            nn.Linear(self.in_channels, self.inter_channels))
        self.W = nn.Linear(self.inter_channels, self.in_channels)

    # def resize(self, x):
    #     b=x.size(0)
    #     fc=nn.Linear(self.in_channels, b).cuda()
    #     mask=fc(x)
    #     mask=self.softmax(mask)
    #     return mask

    def forward(self, x):
        bbox_num = x.size(0)
        x_r = x.view(bbox_num, -1)
        g_x = self.g(x_r)  # 2 , 128 , 150 x 150
        theta_x = self.theta(x_r)  # 2 , 128 , 300 x 300
        phi_x = self.phi(x_r)  # 2 , 128 , 150 x 150
        phi_x = phi_x.permute(1, 0)
        f = torch.matmul(theta_x, phi_x)  # 2 , 300x300 , 150x150
        # mask=self.resize(x_r)
        # f = f*mask
        f_div_C = self.softmax(f)

        y = torch.matmul(f_div_C, g_x)  # 2, 128, 300x300
        W_y = self.W(y)
        z = W_y.view(x.size()) + x

        return z


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
        self.extra_heads = nn.ModuleList([copy.deepcopy(extra_head)
                                         for _ in range(num_extra_head)])

        self.up_channels = nn.ModuleList([nn.ModuleList(
            [nn.Sequential(*([nn.Conv2d(int(256 * self.head.width), int(256 * e.width), kernel_size=1)] +
                             [nn.Conv2d(int(256 * e.width), int(256 * e.width), kernel_size=1) for _ in range(i)])),
             nn.Sequential(*([nn.Conv2d(int(512 * self.head.width), int(512 * e.width), kernel_size=1)] +
                           [nn.Conv2d(int(512 * e.width), int(512 * e.width), kernel_size=1) for _ in range(i)])),
             nn.Sequential(*([nn.Conv2d(int(1024 * self.head.width), int(1024 * e.width), kernel_size=1)] +
                             [nn.Conv2d(int(1024 * e.width), int(1024 * e.width), kernel_size=1) for _ in range(i)]))]
        ) for i, e in enumerate(self.extra_heads)])

        # self.student_non_local = nn.ModuleList(
        #     [
        #         GloRe_Unit_2D(int(256 * extra_head.width), int(256 * extra_head.width)),
        #         GloRe_Unit_2D(int(512 * extra_head.width), int(512 * extra_head.width)),
        #         GloRe_Unit_2D(int(1024 * extra_head.width), int(1024 * extra_head.width)),
        #     ]
        # )
        # self.teacher_non_local = nn.ModuleList(
        #     [
        #         GloRe_Unit_2D(int(256 * extra_head.width), int(256 * extra_head.width)),
        #         GloRe_Unit_2D(int(512 * extra_head.width), int(512 * extra_head.width)),
        #         GloRe_Unit_2D(int(1024 * extra_head.width), int(1024 * extra_head.width)),
        #     ]
        # )
        # self.non_local_adaptation = nn.ModuleList([
        #     nn.Conv2d(int(256 * extra_head.width), int(256 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(int(512 * extra_head.width), int(512 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(int(1024 * extra_head.width), int(1024 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        # ])

        # self.student_relation = nn.ModuleList(
        #     [
        #         ObjectRelation_1(in_channels=int(256 * extra_head.width) * 9),
        #         ObjectRelation_1(in_channels=int(512 * extra_head.width) * 9),
        #         ObjectRelation_1(in_channels=int(1024 * extra_head.width) * 9),
        #     ]
        # )
        # self.teacher_relation = nn.ModuleList(
        #     [
        #         ObjectRelation_1(in_channels=int(256 * extra_head.width) * 9),
        #         ObjectRelation_1(in_channels=int(512 * extra_head.width) * 9),
        #         ObjectRelation_1(in_channels=int(1024 * extra_head.width) * 9),
        #     ]
        # )
        # self.relation_adaptation = nn.ModuleList([
        #     nn.Conv2d(int(256 * extra_head.width), int(256 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(int(512 * extra_head.width), int(512 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(int(1024 * extra_head.width), int(1024 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        # ])
        # self.for_adaptation = nn.ModuleList([
        #     nn.Conv2d(int(256 * extra_head.width), int(256 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(int(512 * extra_head.width), int(512 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(int(1024 * extra_head.width), int(1024 * extra_head.width),
        #               kernel_size=1, stride=1, padding=0),
        # ])
        # self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )

            # with torch.no_grad():
            #     t_fpn_outs = t_model.backbone(x)

            total_extra_loss = 0
            # total_drkd_loss = 0
            for extra_index, extra_head in enumerate(self.extra_heads):
                extra_fpn_outs = list(fpn_outs)
                for i in range(len(extra_fpn_outs)):
                    extra_fpn_outs[i] = self.up_channels[extra_index][i](extra_fpn_outs[i])
                    # flatten_fpn_outs = extra_fpn_outs[i].view(extra_fpn_outs[i].shape[0], -1)
                    # flatten_t_fpn_outs = t_fpn_outs[i].view(t_fpn_outs[i].shape[0], -1)
                    # cosine_loss = 1 - self.cosine_similarity(flatten_fpn_outs, flatten_t_fpn_outs)
                    # total_drkd_loss += torch.sum(cosine_loss) / cosine_loss.shape[0]
                extra_fpn_outs = tuple(extra_fpn_outs)
                extra_loss, extra_iou_loss, extra_conf_loss, extra_cls_loss, extra_l1_loss, extra_num_fg = self.extra_heads[extra_index](
                    extra_fpn_outs, targets, x
                )
                total_extra_loss += extra_loss

                # Caculate drkd loss
                # kd_nonlocal_loss = 0
                # kd_relation_loss = 0
                # kd_foreground_loss = 0
                # for i in range(len(extra_fpn_outs)):
                #     s_relation = self.student_non_local[i](extra_fpn_outs[i])
                #     t_relation = self.teacher_non_local[i](t_fpn_outs[i])
                #     kd_nonlocal_loss += torch.dist(
                #         self.non_local_adaptation[i](s_relation), t_relation, p=2)

                #     mixup = targets.shape[2] > 5
                #     if mixup:
                #         label_cut = targets[..., :5]
                #     else:
                #         label_cut = targets
                #     nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
                #     for batch_idx in range(targets.shape[0]):
                #         num_gt = int(nlabel[batch_idx])
                #         if num_gt == 0:
                #             # print("num_gt == 0")
                #             continue
                #         gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5]
                #         # convert from xywh to tlbr
                #         gt_bboxes_per_image[:, :2] = gt_bboxes_per_image[:, :2] - \
                #             gt_bboxes_per_image[:, 2:] / 2
                #         gt_bboxes_per_image[:, 2:] = gt_bboxes_per_image[:, :2] + gt_bboxes_per_image[:, 2:]
                #         # Normalize for all value > 0
                #         gt_bboxes_per_image = torch.clamp(gt_bboxes_per_image, min=0)

                #         s_region = roi_align(
                #             extra_fpn_outs[i], boxes=[gt_bboxes_per_image], output_size=3, spatial_scale=extra_fpn_outs[i].shape[-1] / 1088)
                #         t_region = roi_align(
                #             t_fpn_outs[i], boxes=[gt_bboxes_per_image], output_size=3, spatial_scale=t_fpn_outs[i].shape[-1] / 1088)
                #         s_object_relation = self.student_relation[i](s_region)
                #         t_object_relation = self.teacher_relation[i](t_region)
                #         kd_relation_loss += torch.dist(self.relation_adaptation[i](
                #             s_object_relation), t_object_relation, p=2)
                #         kd_foreground_loss += torch.dist(self.for_adaptation[i](s_region), t_region, p=2)

                #         kd_nonlocal_loss *= 4e-3
                #         kd_relation_loss *= 0.05
                #         kd_foreground_loss *= 0.06

                # total_drkd_loss += kd_nonlocal_loss + kd_relation_loss + kd_foreground_loss

            outputs = {
                "total_loss": loss + total_extra_loss,
                "loss": loss,
                "extra_loss": total_extra_loss,
                "num_extra_head": len(self.extra_heads),
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
