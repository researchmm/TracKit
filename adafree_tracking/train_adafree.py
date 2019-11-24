# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: SiamFC training script
# ------------------------------------------------------------------------------

import _init_paths
import os
import shutil
import time
import math
import pprint
import argparse
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.utils import build_lr_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import models.models as models
from utils.utils import create_logger, print_speed, load_pretrain, restore_from, save_model
from dataset.adafree import AdaDataset
from core.config import config, update_config
from core.function import ada_train

eps = 1e-5

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train SiamFC')
    # general
    parser.add_argument('--cfg', type=str, default='experiments/train/SiamFC.yaml', help='yaml configure file name')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')

    args = parser.parse_args()

    return args


def reset_config(config, args):
    """
    set gpus and workers
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def check_trainable(model, logger):
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params


def get_optimizer(cfg, trainable_params):
    """
    get optimizer
    """

    optimizer = torch.optim.SGD(trainable_params, cfg.ADAFREE.TRAIN.LR,
                    momentum=cfg.ADAFREE.TRAIN.MOMENTUM,
                    weight_decay=cfg.ADAFREE.TRAIN.WEIGHT_DECAY)

    return optimizer

# from siamrpn++
def build_opt_lr(cfg, model, current_epoch=0):
    # fix all backbone first
    for param in model.features.features.parameters():
        param.requires_grad = False
    for m in model.features.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # for param in model.neck.parameters():
    #     param.requires_grad = False
    # for m in model.neck.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()
    #
    # for param in model.connect_model.parameters():
    #     param.requires_grad = False
    # for m in model.connect_model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()

    if current_epoch >= cfg.ADAFREE.TRAIN.UNFIX_EPOCH:
        if len(cfg.ADAFREE.TRAIN.TRAINABLE_LAYER) > 0:  # specific trainable layers
            for layer in cfg.ADAFREE.TRAIN.TRAINABLE_LAYER:
                for param in getattr(model.features.features, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.features.features, layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
        else:    # train all backbone layers
            for param in model.features.features.parameters():
                param.requires_grad = True
            for m in model.features.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.features.features.parameters():
            param.requires_grad = False
        for m in model.features.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.features.features.parameters()),
                          'lr': cfg.ADAFREE.TRAIN.LAYERS_LR * cfg.ADAFREE.TRAIN.BASE_LR}]
    try:
        trainable_params += [{'params': model.neck.parameters(),
                                  'lr': cfg.ADAFREE.TRAIN.BASE_LR}]
    except:
        pass

    trainable_params += [{'params': model.connect_model.parameters(),
                          'lr': cfg.ADAFREE.TRAIN.BASE_LR}]


    # print trainable parameter (first check)
    print('==========first check trainable==========')
    for param in trainable_params:
        print(param)


    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.ADAFREE.TRAIN.MOMENTUM,
                                weight_decay=cfg.ADAFREE.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, cfg, epochs=cfg.ADAFREE.TRAIN.END_EPOCH)
    lr_scheduler.step(cfg.ADAFREE.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def lr_decay(cfg, optimizer):
    if cfg.ADAFREE.TRAIN.LR_POLICY == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.8685)
    elif cfg.ADAFREE.TRAIN.LR_POLICY == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif cfg.ADAFREE.TRAIN.LR_POLICY == 'Reduce':
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
    elif cfg.ADAFREE.TRAIN.LR_POLICY == 'log':
        scheduler = np.logspace(math.log10(cfg.ADAFREE.TRAIN.LR), math.log10(cfg.ADAFREE.TRAIN.LR_END), cfg.ADAFREE.TRAIN.END_EPOCH)
    else:
        raise ValueError('unsupported learing rate scheduler')

    return scheduler


def pretrain_zoo():
    GDriveIDs = dict()
    GDriveIDs['AdaFree'] = "1UGriYoerXFW48_tf9R1NzwQ06M-5Yz-K"
    return GDriveIDs

def main():
    # [*] args, loggers and tensorboard
    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'ADAFREE', 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # [*] gpus parallel and model prepare
    # prepare pretrained model -- download from google drive
    # auto-download train model from GoogleDrive
    if not os.path.exists('./pretrain'):
        os.makedirs('./pretrain')
    try:
        DRIVEID = pretrain_zoo()

        if not os.path.exists('./pretrain/{}'.format(config.ADAFREE.TRAIN.PRETRAIN)):
            os.system(
                'wget --no-check-certificate \'https://drive.google.com/uc?export=download&id={0}\' -O ./pretrain/{1}'
                .format(DRIVEID[config.ADAFREE.TRAIN.MODEL], config.ADAFREE.TRAIN.PRETRAIN))
    except:
        print('auto-download pretrained model fail, please download it and put it in pretrain directory')


    model = models.__dict__[config.ADAFREE.TRAIN.MODEL]()  # build model
    print(model)
    try:
        model = load_pretrain(model, './pretrain/{0}'.format(config.ADAFREE.TRAIN.PRETRAIN))    # load pretrain
    except:
        print('=============train from scrach================')

    # get optimizer
    if not config.ADAFREE.TRAIN.START_EPOCH == config.ADAFREE.TRAIN.UNFIX_EPOCH:
        optimizer, lr_scheduler = build_opt_lr(config, model, config.ADAFREE.TRAIN.START_EPOCH)
    else:
        optimizer, lr_scheduler = build_opt_lr(config, model, 0)  # resume wrong (last line)

    # check trainable again
    print('==========double check trainable==========')
    trainable_params = check_trainable(model, logger)           # print trainable params info

    if config.ADAFREE.TRAIN.RESUME and config.ADAFREE.TRAIN.START_EPOCH != 0:   # resume
        model.features.unfix((config.ADAFREE.TRAIN.START_EPOCH - 1) / config.ADAFREE.TRAIN.END_EPOCH)
        model, optimizer, args.start_epoch, arch = restore_from(model, optimizer, config.ADAFREE.TRAIN.RESUME)

    # parallel
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))

    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)
    # try to solve gpu problem
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.GPUS])
    logger.info(lr_scheduler)
    logger.info('model prepare done')

    # [*] train

    for epoch in range(config.ADAFREE.TRAIN.START_EPOCH, config.ADAFREE.TRAIN.END_EPOCH):
        # build dataloader, benefit to tracking
        train_set = AdaDataset(config)
        train_loader = DataLoader(train_set, batch_size=config.ADAFREE.TRAIN.BATCH * gpu_num, num_workers=config.WORKERS, pin_memory=True, sampler=None, drop_last=True)

        # check if it's time to train backbone
        if epoch == config.ADAFREE.TRAIN.UNFIX_EPOCH:
            logger.info('training backbone')
            optimizer, lr_scheduler = build_opt_lr(config, model.module, epoch)
            print('==========double check trainable==========')
            check_trainable(model, logger)  # print trainable params info

        lr_scheduler.step(epoch)
        curLR = lr_scheduler.get_cur_lr()


        model, writer_dict = ada_train(train_loader, model, optimizer, epoch + 1, curLR, config, writer_dict, logger, device=device)

        # save model
        save_model(model, epoch, optimizer, config.ADAFREE.TRAIN.MODEL, config, isbest=False)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()




