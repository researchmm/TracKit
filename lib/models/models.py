import math
import torch
import torch.nn as nn
from .ocean import Ocean_
from .oceanplus import OceanPlus_
from .oceanTRT import OceanTRT_
from .siamfc import SiamFC_
from .connect import box_tower, AdjustLayer, AlignHead, Corr_Up, MultiDiCorr, OceanCorr
from .backbones import ResNet50, ResNet22W
from .mask import MMS, MSS
from .modules import MultiFeatureBase

import os
import sys
sys.path.append('../lib')

import models.online.classifier.features as clf_features
import models.online.classifier.initializer as clf_initializer
import models.online.classifier.optimizer as clf_optimizer
import models.online.classifier.linear_filter as target_clf
from online import TensorList, load_network


class Ocean(Ocean_):
    def __init__(self, align=False, online=False):
        super(Ocean, self).__init__()
        self.features = ResNet50(used_layers=[3], online=online)   # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model = box_tower(inchannels=256, outchannels=256, towernum=4)
        self.align_head = AlignHead(256, 256) if align else None


class OceanTRT(OceanTRT_):
    def __init__(self, online=False, align=False):
        super(OceanTRT, self).__init__()
        self.features = ResNet50(used_layers=[3], online=online)  # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model0 = MultiDiCorr(inchannels=256, outchannels=256)
        self.connect_model1 = box_tower(inchannels=256, outchannels=256, towernum=4)
        self.connect_model2 = OceanCorr()


class OceanPlus(OceanPlus_):
    def __init__(self, online=False, mms=False):
        super(OceanPlus, self).__init__()
        self.features = ResNet50(used_layers=[3], online=online)   # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model = box_tower(inchannels=256, outchannels=256, towernum=4)

        if mms:
            self.mask_model = MMS()
        else:
            self.mask_model = MSS()


#class OceanPlusTRT(OceanPlusTRT_):
#    def __init__(self, online=False):
#        super(OceanPlusTRT, self).__init__()
#        self.features = ResNet50(used_layers=[3], online=online)  # in param
#        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
#        self.connect_model0 = MultiDiCorr(inchannels=256, outchannels=256)
#        self.connect_model1 = box_tower(inchannels=256, outchannels=256, towernum=4)
#        self.connect_model2 = OceanCorr()
#        self.mask_model = MultiRefineTRT(addCorr=True, mulOradd='add')


# ------------------------------
# SiamDW in CVPR2019
# ------------------------------
class SiamDW(SiamFC_):
    def __init__(self, **kwargs):
        """
        only SiamDW here
        """
        super(SiamDW, self).__init__(**kwargs)
        self.features = ResNet22W()
        self.connect_model = Corr_Up()

# ================================
# Some functions for online model
# ================================
class OninleRes18(MultiFeatureBase):
    """
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_layers = ['layer3']


    def initialize(self, siam_net):

        self.net = siam_net

        self.layer_stride = {'conv1': 2, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32, 'classification': 16, 'fc': None}
        self.layer_dim = {'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'classification': 256,'fc': None}

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1]*len(self.output_layers)

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net
        if hasattr(self, 'iou_predictor'):
            del self.iou_predictor
        if hasattr(self, 'iounet_backbone_features'):
            del self.iounet_backbone_features
        if hasattr(self, 'iounet_features'):
            del self.iounet_features

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])

    def extract(self, im: torch.Tensor):
        # im = im/255       # remove this for siam_net
        # im -= self.mean
        # im /= self.std
        im = im.cuda()

        with torch.no_grad():
            output_features = self.net.extract_for_online(im)

        return TensorList([output_features])

class NetWrapper:
    """Used for wrapping networks in pytracking.
    Network modules and functions can be accessed directly as if they were members of this class."""
    _rec_iter=0
    def __init__(self, net_path, use_gpu=True):
        self.net_path = net_path
        self.use_gpu = use_gpu
        self.net = None

    def __getattr__(self, name):
        if self._rec_iter > 0:
            self._rec_iter = 0
            return None
        self._rec_iter += 1
        try:
            ret_val = getattr(self.net, name)
        except Exception as e:
            self._rec_iter = 0
            raise e
        self._rec_iter = 0
        return ret_val

    def load_network(self):
        self.net = load_network(self.net_path)
        if self.use_gpu:
            self.cuda()
        self.eval()

    def initialize(self):
        self.load_network()

class NetWithBackbone(NetWrapper):
    """Wraps a network with a common backbone.
    Assumes the network have a 'extract_backbone_features(image)' function."""
    def initialize(self, siam_net):
        super().initialize()
        self._mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self._std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        self.net = siam_net

    def preprocess_image(self, im: torch.Tensor):
        """Normalize the image with the mean and standard deviation used by the network."""

        im = im/255
        im -= self._mean
        im /= self._std

        im = im.cuda()

        return im

    def extract_backbone(self, im: torch.Tensor):
        """Extract backbone features from the network.
        Expects a float tensor image with pixel range [0, 255]."""
        im = self.preprocess_image(im)
        return self.net.extract_for_online(im)


class ONLINEnet(nn.Module):
    """The ONLINE network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression.
        train_feature_extractor:  Whether feature extractor should be trained or not."""

    def __init__(self, feature_extractor, classifier, classification_layer, train_feature_extractor=True):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        # self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))
        self.output_layers = sorted(list(set(self.classification_layer)))

        self._mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self._std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        if not train_feature_extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the ONLINE network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # TODO: DEBUG REMOVE HERE
        train_imgs = torch.ones_like(train_imgs)
        test_imgs = torch.ones_like(test_imgs)
        train_bb = torch.ones_like(train_bb)
        test_proposals = torch.ones_like(test_proposals)


        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        # train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        # test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        # iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        # return target_scores, iou_pred
        return target_scores

    # def get_backbone_clf_feat(self, backbone_feat):
    #     # feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
    #     feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})  # zzp
    #     if len(self.classification_layer) == 1:
    #         return feat[self.classification_layer[0]]
    #     return feat

    def get_backbone_clf_feat(self, backbone_feat):  # zzp
        return backbone_feat

    # def get_backbone_bbreg_feat(self, backbone_feat):
    #     return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor.extract_for_online(im)  # zzp

    def extract_features(self, im, layers=None):
        if layers is None:
            # layers = self.bb_regressor_layer + ['classification']
            layers = ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})

    def preprocess_image(self, im: torch.Tensor):
        """Normalize the image with the mean and standard deviation used by the network."""

        im = im/255
        im -= self._mean
        im /= self._std

        im = im.cuda()

        return im

    def extract_backbone(self, im: torch.Tensor):
        """Extract backbone features from the network.
        Expects a float tensor image with pixel range [0, 255]."""
        im = self.preprocess_image(im)
        return self.extract_backbone_features(im)


def ONLINEnet50(filter_size=4, optim_iter=5, optim_init_step=0.9, optim_init_reg=0.1,
              classification_layer='layer3', feat_stride=16, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=0.85, num_dist_bins=100, bin_displacement=0.1,
              mask_init_factor=3.0, score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), backbone=None):
    # Backbone
    backbone_net = backbone   # siamnet

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the  classifier
    optimizer = clf_optimizer.ONLINESteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # ONLINE network
    net = ONLINEnet(feature_extractor=backbone_net, classifier=classifier, classification_layer=classification_layer)
    return net
