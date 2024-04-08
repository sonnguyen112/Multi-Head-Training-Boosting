#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import torch
from .GloRe import GloRe_Unit_2D
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
        x_r=x.view(bbox_num, -1) 
        g_x = self.g(x_r)   #   2 , 128 , 150 x 150
        theta_x = self.theta(x_r)   #   2 , 128 , 300 x 300
        phi_x = self.phi(x_r)      #   2 , 128 , 150 x 150
        phi_x = phi_x.permute(1, 0) 
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        # mask=self.resize(x_r)
        # f = f*mask
        f_div_C = self.softmax(f)

        y = torch.matmul(f_div_C, g_x)  #   2, 128, 300x300
        W_y = self.W(y)
        z = W_y.view(x.size()) + x

        return z


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
        self.teacher_non_local = nn.ModuleList(
            [
                GloRe_Unit_2D(320,320),
                GloRe_Unit_2D(640,640),
                GloRe_Unit_2D(1280,1280),
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1280, kernel_size=1, stride=1, padding=0),
        ])
        self.for_adaptation = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1280, kernel_size=1, stride=1, padding=0),
        ])
        self.student_relation = nn.ModuleList(
            [
                ObjectRelation_1(in_channels=128 * 9),
                ObjectRelation_1(in_channels=256 * 9),
                ObjectRelation_1(in_channels=512 * 9),
            ]
        )
        self.teacher_relation = nn.ModuleList(
            [
                ObjectRelation_1(in_channels=320 * 9),
                ObjectRelation_1(in_channels=640 * 9),
                ObjectRelation_1(in_channels=1280 * 9),
            ]
        )
        self.relation_adaptation = nn.ModuleList([
            nn.Conv2d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 640, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1280, kernel_size=1, stride=1, padding=0),
        ])


    def forward(self, x, targets=None, t_model = None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            print(targets.shape)
            mixup = targets.shape[2] > 5
            if mixup:
                label_cut = targets[..., :5]
            else:
                label_cut = targets
            nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
            print(nlabel)
            batch_gt_bboxes = []
            for batch_idx in range(targets.shape[0]):
                num_gt = int(nlabel[batch_idx])
                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5]
                # convert from xywh to tlbr
                gt_bboxes_per_image[:, :2] = gt_bboxes_per_image[:, :2] - gt_bboxes_per_image[:, 2:] / 2
                gt_bboxes_per_image[:, 2:] = gt_bboxes_per_image[:, :2] + gt_bboxes_per_image[:, 2:]
                # Normalize for all value > 0
                gt_bboxes_per_image = torch.clamp(gt_bboxes_per_image, min=0)
                batch_gt_bboxes.append(gt_bboxes_per_image)
            
            kd_nonlocal_loss = 0
            kd_foreground_loss = 0
            kd_relation_loss = 0

            t_feat = t_model.backbone(x)

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

                s_region=roi_align(student_feature, boxes=batch_gt_bboxes, output_size=3, spatial_scale=student_feature.shape[-1] / 1440)
                t_region=roi_align(teacher_feature, boxes=batch_gt_bboxes, output_size=3, spatial_scale=teacher_feature.shape[-1] / 1440)
                s_object_relation = self.student_relation[i](s_region)
                t_object_relation = self.teacher_relation[i](t_region)
                print(s_object_relation.shape)
                exit()
                kd_relation_loss += torch.dist(self.relation_adaptation[i](s_object_relation), t_object_relation, p=2)


            kd_nonlocal_loss *= 0.004
            kd_foreground_loss *= 0.006
            kd_relation_loss *= 0.005

                
            outputs = {
                "total_loss": loss + kd_foreground_loss + kd_nonlocal_loss + kd_relation_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "kd_foreground_loss": kd_foreground_loss,
                "kd_nonlocal_loss": kd_nonlocal_loss,
                "kd_relation_loss": kd_relation_loss
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
