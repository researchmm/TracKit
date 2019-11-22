# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: Models for AdaFree
# Main Results: see readme.md
# ------------------------------------------------------------------------------

from .adafree import AdaFree_
from .connect import box_tower, AdjustLayer, AlignHead
from .backbones import ResNet50


import torch
import torchvision
from .modules import MultiFeatureBase

import os
import sys
sys.path.append('../lib')
# from atom import TensorList, load_network


class AdaFree(AdaFree_):
    def __init__(self, align=True):
        super(AdaFree, self).__init__()
        self.features = ResNet50(used_layers=[3])   # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model = box_tower(inchannels=256, outchannels=256, towernum=4)
        self.align_head = AlignHead(256, 256) if align else None





# =======================================================
# ATOM series (Different structure with Siamese Seriese)
# =======================================================
# class ATOMRes18(MultiFeatureBase):
#     """ResNet18 feature with the ATOM IoUNet.
#     args:
#         output_layers: List of layers to output.
#         net_path: Relative or absolute net path (default should be fine).
#     """
#     def __init__(self, output_layers=('layer3',), pretrained_path='./snapshot', constructor_fun_name = 'atom_resnet18', constructor_module = 'lib.models.atom.bbreg.atom', *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.output_layers = list(output_layers)
#         self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#         self.ppath = pretrained_path
#         self.fun_name = constructor_fun_name
#         self.module = constructor_module
#
#
#     def initialize(self):
#
#         if not self.ppath or not os.path.exists(self.ppath):
#             raise Exception('========= pretrained model not found =========')
#
#         self.net = load_network(self.ppath, self.fun_name, self.module)
#
#         self.net.cuda()
#         self.net.eval()
#
#         self.iou_predictor = self.net.bb_regressor
#
#         self.layer_stride = {'conv1': 2, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32, 'classification': 16, 'fc': None}
#         self.layer_dim = {'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'classification': 256,'fc': None}
#
#         self.iounet_feature_layers = self.net.bb_regressor_layer
#
#         if isinstance(self.pool_stride, int) and self.pool_stride == 1:
#             self.pool_stride = [1]*len(self.output_layers)
#
#         self.feature_layers = sorted(list(set(self.output_layers + self.iounet_feature_layers)))
#
#         self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
#         self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)
#
#     def free_memory(self):
#         if hasattr(self, 'net'):
#             del self.net
#         if hasattr(self, 'iou_predictor'):
#             del self.iou_predictor
#         if hasattr(self, 'iounet_backbone_features'):
#             del self.iounet_backbone_features
#         if hasattr(self, 'iounet_features'):
#             del self.iounet_features
#
#     def dim(self):
#         return TensorList([self.layer_dim[l] for l in self.output_layers])
#
#     def stride(self):
#         return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])
#
#     def extract(self, im: torch.Tensor):
#         im = im/255
#         im -= self.mean
#         im /= self.std
#         im = im.cuda()
#
#         with torch.no_grad():
#             output_features = self.net.extract_features(im, self.feature_layers)
#
#         # Store the raw resnet features which are input to iounet
#         self.iounet_backbone_features = TensorList([output_features[layer].clone() for layer in self.iounet_feature_layers])
#
#         # Store the processed features from iounet, just before pooling
#         with torch.no_grad():
#             self.iounet_features = TensorList(self.iou_predictor.get_iou_feat(self.iounet_backbone_features))
#
#         return TensorList([output_features[layer] for layer in self.output_layers])


