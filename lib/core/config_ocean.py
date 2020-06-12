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

config.OCEAN = edict()
config.OCEAN.TRAIN = edict()
config.OCEAN.TEST = edict()
config.OCEAN.TUNE = edict()
config.OCEAN.DATASET = edict()
config.OCEAN.DATASET.VID = edict()
config.OCEAN.DATASET.GOT10K = edict()
config.OCEAN.DATASET.COCO = edict()
config.OCEAN.DATASET.DET = edict()
config.OCEAN.DATASET.LASOT = edict()
config.OCEAN.DATASET.YTB = edict()
config.OCEAN.DATASET.VISDRONE = edict()

# augmentation
config.OCEAN.DATASET.SHIFT = 4
config.OCEAN.DATASET.SCALE = 0.05
config.OCEAN.DATASET.COLOR = 1
config.OCEAN.DATASET.FLIP = 0
config.OCEAN.DATASET.BLUR = 0
config.OCEAN.DATASET.GRAY = 0
config.OCEAN.DATASET.MIXUP = 0
config.OCEAN.DATASET.CUTOUT = 0
config.OCEAN.DATASET.CHANNEL6 = 0
config.OCEAN.DATASET.LABELSMOOTH = 0
config.OCEAN.DATASET.ROTATION = 0
config.OCEAN.DATASET.SHIFTs = 64
config.OCEAN.DATASET.SCALEs = 0.18

# vid
config.OCEAN.DATASET.VID.PATH = '$data_path/vid/crop511'
config.OCEAN.DATASET.VID.ANNOTATION = '$data_path/vid/train.json'

# got10k
config.OCEAN.DATASET.GOT10K.PATH = '$data_path/got10k/crop511'
config.OCEAN.DATASET.GOT10K.ANNOTATION = '$data_path/got10k/train.json'
config.OCEAN.DATASET.GOT10K.RANGE = 100
config.OCEAN.DATASET.GOT10K.USE = 200000

# visdrone
config.OCEAN.DATASET.VISDRONE.ANNOTATION = '$data_path/visdrone/train.json'
config.OCEAN.DATASET.VISDRONE.PATH = '$data_path/visdrone/crop271'
config.OCEAN.DATASET.VISDRONE.RANGE = 100
config.OCEAN.DATASET.VISDRONE.USE = 100000

# train
config.OCEAN.TRAIN.GROUP = "resrchvc"
config.OCEAN.TRAIN.EXID = "setting1"
config.OCEAN.TRAIN.MODEL = "Ocean"
config.OCEAN.TRAIN.RESUME = False
config.OCEAN.TRAIN.START_EPOCH = 0
config.OCEAN.TRAIN.END_EPOCH = 50
config.OCEAN.TRAIN.TEMPLATE_SIZE = 127
config.OCEAN.TRAIN.SEARCH_SIZE = 255
config.OCEAN.TRAIN.STRIDE = 8
config.OCEAN.TRAIN.BATCH = 32
config.OCEAN.TRAIN.PRETRAIN = 'pretrain.model'
config.OCEAN.TRAIN.LR_POLICY = 'log'
config.OCEAN.TRAIN.LR = 0.001
config.OCEAN.TRAIN.LR_END = 0.00001
config.OCEAN.TRAIN.MOMENTUM = 0.9
config.OCEAN.TRAIN.WEIGHT_DECAY = 0.0001
config.OCEAN.TRAIN.WHICH_USE = ['GOT10K']  # VID or 'GOT10K'

# test
config.OCEAN.TEST.MODEL = config.OCEAN.TRAIN.MODEL
config.OCEAN.TEST.DATA = 'VOT2019'
config.OCEAN.TEST.START_EPOCH = 30
config.OCEAN.TEST.END_EPOCH = 50

# tune
config.OCEAN.TUNE.MODEL = config.OCEAN.TRAIN.MODEL
config.OCEAN.TUNE.DATA = 'VOT2019'
config.OCEAN.TUNE.METHOD = 'TPE'  # 'GENE' or 'RAY'



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
        if model_name not in ['OCEAN', 'SIAMRPN']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k=OCEAN or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
