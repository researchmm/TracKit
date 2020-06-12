# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

class OceanTRT_(nn.Module):
    def __init__(self):
        super(OceanTRT_, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None  # for online tracking
        self.neck = None
        self.search_size = 255
        self.score_size = 25


    def tensorrt_init(self, trt_net, corr=None):
        """
        TensorRT init
        """
        self.t_backbone255, self.s_backbone_siam255, self.s_backbone_siam287, self.s_backbone_online, self.t_neck255, \
        self.s_neck255, self.s_neck287, self.multiDiCorr255, self.multiDiCorr287, self.boxtower255, self.boxtower287 = trt_net

        if corr:
            self.multiDiCorr255, self.multiDiCorr287 = corr

    def extract_for_online(self, x):
        xf = self.s_backbone_online(x, torch.Tensor([1]).cuda())
        return xf

    def template(self, z):
        _, _, _, self.zf = self.t_backbone255(z, torch.Tensor([]).cuda())
        self.zf_ori = self.t_neck255(self.zf)
        self.zf = self.zf_ori[:, :, 4:-4, 4:-4].contiguous()

    def track(self, x):
        """
        Please see OceanOnlinePT for pytorch version (more clean)
        """
        b1, b2, b3, xf = self.s_backbone_siam255(x, torch.Tensor([]).cuda())
        xf = self.s_neck255(xf)  # b4

        # backbone encode (something is wrong with connect model)
        cls_z0, cls_z1, cls_z2, cls_x0, cls_x1, cls_x2, reg_z0, reg_z1, reg_z2, reg_x0, reg_x1, reg_x2 = self.multiDiCorr255(xf, self.zf)

        # correlation
        cls_z = [cls_z0, cls_z1, cls_z2]
        cls_x = [cls_x0, cls_x1, cls_x2]
        reg_z = [reg_z0, reg_z1, reg_z2]
        reg_x = [reg_x0, reg_x1, reg_x2]

        cls_dw, reg_dw = self.connect_model2(cls_z, cls_x, reg_z, reg_x)
        # cls and reg
        bbox_pred, cls_pred = self.boxtower255(cls_dw, reg_dw)

        return cls_pred, bbox_pred.squeeze(0)
