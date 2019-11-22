# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: This script provides epoch-train for siamfc and siamrpn
# ------------------------------------------------------------------------------

import math
import time
import torch
from utils.utils import print_speed


def ada_train(train_loader, model,  optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].to(device)
        search = input[1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[3].float().to(device)
        reg_weight = input[4].float().to(device)

        cls_loss_ori, cls_loss_align, reg_loss = model(template, search, label_cls, reg_target=reg_label, reg_weight=reg_weight)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        if cls_loss_align is not None:
            cls_loss_align = torch.mean(cls_loss_align)
            loss = cls_loss_ori + cls_loss_align + 1.2 * reg_loss
        else:
            cls_loss_align = 0
            loss = cls_loss_ori +  1.2 * reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.ADAFREE.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict



def unfix_more(model, optimizer, epoch, cfg, cur_lr, logger):
    if model.module.features.unfix(epoch / cfg.ADAFREE.TRAIN.END_EPOCH):
        logger.info('unfix part model.')
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, cur_lr,
                                    momentum=cfg.ADAFREE.TRAIN.MOMENTUM,
                                    weight_decay=cfg.ADAFREE.TRAIN.WEIGHT_DECAY)

        logger.info('trainable params:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)
    else:
        logger.info('no unfix in this epoch.')

    return model, optimizer


def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
