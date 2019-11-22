# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: This script provides CIR backbones proposed in CVPR2019 paper
# Main Results: see readme.md
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from .modules import  Bottleneck, ResNet_plus2

eps = 1e-5
# ---------------------
# RPN++
# ---------------------
class ResNet50(nn.Module):
    """
    RPN++ model
    """
    def __init__(self, used_layers=[2, 3, 4]):
        super(ResNet50, self).__init__()
        self.features = ResNet_plus2(Bottleneck, [3, 4, 6, 3], used_layers=used_layers)
        # self.unfix(0.0)

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == '__main__':
    import torch
    net = ResNet50().cuda()
    print(net)

    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total params" + str(k/1e6) + "M")

    search = torch.rand(1, 3, 255, 255)
    search = torch.Tensor(search).cuda()
    out = net(search)
    print(out.size())

    print()
