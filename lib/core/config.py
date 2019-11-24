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
config.GPUS = "0,1,2,3"
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
config.ADAFREE.TRAIN.EXID = "vot19_ex5_setting_default"
config.ADAFREE.TRAIN.MODEL = "SiamFCRes22W"
config.ADAFREE.TRAIN.RESUME = False
config.ADAFREE.TRAIN.START_EPOCH = 0
config.ADAFREE.TRAIN.END_EPOCH = 50
config.ADAFREE.TRAIN.TEMPLATE_SIZE = 127
config.ADAFREE.TRAIN.SEARCH_SIZE = 255
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
        if model_name not in ['ADAFREE', 'SIAMRPN']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k=ADAFREE or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
