# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from .modules import  Bottleneck, ResNet_plus2, Bottleneck_BIG_CI, ResNet

eps = 1e-5
# ---------------------
# For Ocean and Ocean+
# ---------------------
class ResNet50(nn.Module):
    def __init__(self, used_layers=[2, 3, 4], online=False):
        super(ResNet50, self).__init__()
        self.features = ResNet_plus2(Bottleneck, [3, 4, 6, 3], used_layers=used_layers, online=online)

    def forward(self, x, online=False):
        if not online:
            x_stages, x = self.features(x, online=online)
            return x_stages, x
        else:
            x = self.features(x, online=online)
            return x

# ---------------------
# For SiamDW
# ---------------------
class ResNet22W(nn.Module):
    """
    ResNet22W: double 3*3 layer (only) channels in residual blob
    """
    def __init__(self):
        super(ResNet22W, self).__init__()
        self.features = ResNet(Bottleneck_BIG_CI, [3, 4], [True, False], [False, True], firstchannels=64, channels=[64, 128])
        self.feature_size = 512

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
