# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: This script provides configs for siamfc and siamrpn
# ------------------------------------------------------------------------------

import os
import yaml
from easydict import EasyDict as edict

config = edict()

# ------config for general parameters------
config.GPUS = "0"
config.WORKERS = 32
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'logs'
config.CHECKPOINT_DIR = 'snapshot'

# #-----————- config for siamfc ------------
config.ADAFREE = edict()
config.ADAFREE.TRAIN = edict()
config.ADAFREE.TEST = edict()
config.ADAFREE.TUNE = edict()
config.ADAFREE.DATASET = edict()
config.ADAFREE.DATASET.VID = edict()       # paper utlized but not recommended
config.ADAFREE.DATASET.GOT10K = edict()    # not utlized in paper but recommended, better performance and more stable
config.ADAFREE.DATASET.COCO = edict()
config.ADAFREE.DATASET.DET = edict()
config.ADAFREE.DATASET.LASOT = edict()
config.ADAFREE.DATASET.YTB = edict()
config.ADAFREE.DATASET.VISDRONE = edict()

# augmentation
config.ADAFREE.DATASET.SHIFT = 4
config.ADAFREE.DATASET.SCALE = 0.05
config.ADAFREE.DATASET.COLOR = 1
config.ADAFREE.DATASET.FLIP = 0
config.ADAFREE.DATASET.BLUR = 0
config.ADAFREE.DATASET.GRAY = 0
config.ADAFREE.DATASET.MIXUP = 0
config.ADAFREE.DATASET.CUTOUT = 0
config.ADAFREE.DATASET.CHANNEL6 = 0
config.ADAFREE.DATASET.LABELSMOOTH = 0
config.ADAFREE.DATASET.ROTATION = 0
config.ADAFREE.DATASET.SHIFTs = 64
config.ADAFREE.DATASET.SCALEs = 0.18

# vid
config.ADAFREE.DATASET.VID.PATH = '/home/zhbli/Dataset/data2/vid/crop511'
config.ADAFREE.DATASET.VID.ANNOTATION = '/home/zhbli/Dataset/data2/vid/train.json'

# got10k
config.ADAFREE.DATASET.GOT10K.PATH = '/data/share/LARGESIAM/got10k/crop511'
config.ADAFREE.DATASET.GOT10K.ANNOTATION = '/data/share/LARGESIAM/got10k/train.json'
config.ADAFREE.DATASET.GOT10K.RANGE = 100
config.ADAFREE.DATASET.GOT10K.USE = 200000

# visdrone
config.ADAFREE.DATASET.VISDRONE.ANNOTATION = '/data2/SMALLSIAM/visdrone/train.json'
config.ADAFREE.DATASET.VISDRONE.PATH = '/data2/SMALLSIAM/visdrone/crop271'
config.ADAFREE.DATASET.VISDRONE.RANGE = 100
config.ADAFREE.DATASET.VISDRONE.USE = 100000

# train
config.ADAFREE.TRAIN.GROUP = "resrchvc"
config.ADAFREE.TRAIN.MODEL = "SiamFCRes22W"
config.ADAFREE.TRAIN.RESUME = False
config.ADAFREE.TRAIN.START_EPOCH = 0
config.ADAFREE.TRAIN.END_EPOCH = 50
config.ADAFREE.TRAIN.TEMPLATE_SIZE = 127
config.ADAFREE.TRAIN.SEARCH_SIZE = 143
config.ADAFREE.TRAIN.STRIDE = 8
config.ADAFREE.TRAIN.BATCH = 32
config.ADAFREE.TRAIN.PRETRAIN = 'resnet23_inlayer.model'
config.ADAFREE.TRAIN.LR_POLICY = 'log'
config.ADAFREE.TRAIN.LR = 0.001
config.ADAFREE.TRAIN.LR_END = 0.0000001
config.ADAFREE.TRAIN.MOMENTUM = 0.9
config.ADAFREE.TRAIN.WEIGHT_DECAY = 0.0001
config.ADAFREE.TRAIN.WHICH_USE = ['GOT10K']  # VID or 'GOT10K'

# test
config.ADAFREE.TEST.MODEL = config.ADAFREE.TRAIN.MODEL
config.ADAFREE.TEST.DATA = 'VOT2015'
config.ADAFREE.TEST.START_EPOCH = 30
config.ADAFREE.TEST.END_EPOCH = 50

# tune
config.ADAFREE.TUNE.MODEL = config.ADAFREE.TRAIN.MODEL
config.ADAFREE.TUNE.DATA = 'VOT2015'
config.ADAFREE.TUNE.METHOD = 'TPE'  # 'GENE' or 'RAY'

# #-----————- config for freemask ------------
config.FREEMASK = edict()
config.FREEMASK.TRAIN = edict()
config.FREEMASK.TEST = edict()
config.FREEMASK.TUNE = edict()
config.FREEMASK.DATASET = edict()
config.FREEMASK.DATASET.YTBVOS = edict()
config.FREEMASK.DATASET.COCO = edict()


# augmentation
config.FREEMASK.DATASET.SHIFT = 4
config.FREEMASK.DATASET.SCALE = 0.05
config.FREEMASK.DATASET.COLOR = 1
config.FREEMASK.DATASET.FLIP = 0
config.FREEMASK.DATASET.BLUR = 0.18
config.FREEMASK.DATASET.GRAY = 0
config.FREEMASK.DATASET.MIXUP = 0
config.FREEMASK.DATASET.CUTOUT = 0
config.FREEMASK.DATASET.CHANNEL6 = 0
config.FREEMASK.DATASET.LABELSMOOTH = 0
config.FREEMASK.DATASET.ROTATION = 0
config.FREEMASK.DATASET.SHIFTs = 0
config.FREEMASK.DATASET.SCALEs = 0
config.FREEMASK.DATASET.TEMPLATE_SMALL = False

# got10k
config.FREEMASK.DATASET.YTBVOS.PATH = '/data/home/v-zhipeng/data/segmentation/Crop/YTBVOS/crop511'
config.FREEMASK.DATASET.YTBVOS.ANNOTATION = '/data/home/v-zhipeng/data/segmentation/Crop/YTBVOS/train.json'
config.FREEMASK.DATASET.YTBVOS.RANGE = 20
config.FREEMASK.DATASET.YTBVOS.USE = 100000

# visdrone
config.FREEMASK.DATASET.COCO.ANNOTATION = '/data/home/v-zhipeng/data/segmentation/Crop/coco/train2017.json'
config.FREEMASK.DATASET.COCO.PATH = '/data/home/v-zhipeng/data/segmentation/Crop/coco/crop511'
config.FREEMASK.DATASET.COCO.RANGE = 1
config.FREEMASK.DATASET.COCO.USE = 100000

# train
config.FREEMASK.TRAIN.MODEL = "FreeDepthMASK"
config.FREEMASK.TRAIN.RESUME = False
config.FREEMASK.TRAIN.START_EPOCH = 0
config.FREEMASK.TRAIN.END_EPOCH = 50
config.FREEMASK.TRAIN.BASE_SIZE = 0
config.FREEMASK.TRAIN.CROP_SIZE = 0
config.FREEMASK.TRAIN.ORIGIN_SIZE = 127
config.FREEMASK.TRAIN.TEMPLATE_SIZE = 127
config.FREEMASK.TRAIN.SEARCH_SIZE = 255
config.FREEMASK.TRAIN.STRIDE = 8
config.FREEMASK.TRAIN.BATCH = 32
config.FREEMASK.TRAIN.PRETRAIN = ''
config.FREEMASK.TRAIN.LR_POLICY = 'log'
config.FREEMASK.TRAIN.LR = 0.001
config.FREEMASK.TRAIN.LR_END = 0.0000001
config.FREEMASK.TRAIN.MOMENTUM = 0.9
config.FREEMASK.TRAIN.WEIGHT_DECAY = 0.0001
config.FREEMASK.TRAIN.WHICH_USE = ['COCO', 'YTBVOS']  # VID or 'GOT10K'

# test
config.FREEMASK.TEST.MODEL = config.FREEMASK.TRAIN.MODEL
config.FREEMASK.TEST.DATA = 'VOT2015'
config.FREEMASK.TEST.START_EPOCH = 30
config.FREEMASK.TEST.END_EPOCH = 50

# tune
config.FREEMASK.TUNE.MODEL = config.FREEMASK.TRAIN.MODEL
config.FREEMASK.TUNE.DATA = 'VOT2015'
config.FREEMASK.TUNE.METHOD = 'TPE'  # 'GENE' or 'RAY'

def _update_dict(k, v, model_name):
    if k in ['TRAIN', 'TEST', 'TUNE']:
        for vk, vv in v.items():
            config[model_name][k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'COCO', 'DET', 'YTB', 'LASOT']:
                config[model_name][k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    try:
                        config[model_name][k][vk][vvk] = vvv
                    except:
                        config[model_name][k][vk] = edict()
                        config[model_name][k][vk][vvk] = vvv

    else:
        config[k] = v   # gpu et.


def update_config(config_file):
    """
    ADD new keys to config
    """
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        model_name = list(exp_config.keys())[0]
        if model_name not in ['ADAFREE', 'FREEMASK']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k=ADAFREE or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
