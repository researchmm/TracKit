import _init_paths
import vot
import os
import cv2
import sys
import random
import argparse
import numpy as np
import torch
import models.models as models

from os.path import exists, join, dirname, realpath
from tracker.oceanplus import OceanPlus
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

from vot import Rectangle,Polygon, Point


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

def rect_from_mask(mask):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))

    w = x1 - x0 + 1
    h = y1 - y0 + 1
    # return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    return [x0 + w/2 , y0 + h/2, w, h]

def mask_from_rect(rect, output_sz):
    '''
    create a binary mask from a given rectangle
    rect: axis-aligned rectangle [x0, y0, width, height]
    output_sz: size of the output [width, height]
    '''
    mask = np.zeros((output_sz[1], output_sz[0]), dtype=np.uint8)
    x0 = max(int(round(rect[0])), 0)
    y0 = max(int(round(rect[1])), 0)
    x1 = min(int(round(rect[0] + rect[2])), output_sz[0])
    y1 = min(int(round(rect[1] + rect[3])), output_sz[1])
    mask[y0:y1, x0:x1] = 1
    return mask

# define tracker
info = edict()
info.arch = "OceanPlus"
info.dataset = "VOT2020"
info.epoch_test = False
info.align = False
info.online = False
mask_vot = True

net = models.__dict__[info.arch](online=info.online, mms=False)
net = load_pretrain(net, "$tracker_path/snapshot/OceanPlusMSS.pth")
net.eval()
net = net.cuda()

# warm up
print('==== warm up ====')
for i in range(10):
    net.template(torch.rand(1, 3, 127, 127).cuda(), torch.rand(1, 127, 127).cuda())
    net.track(torch.rand(1, 3, 255, 255).cuda())

tracker = OceanPlus(info)

# vot2020 settings

if mask_vot:
    handle = vot.VOT("mask")
else:
    handle = vot.VOT("rectangle")

image_file = handle.frame()

if not image_file:
    sys.exit(0)

im = cv2.imread(image_file)  # HxWxC

if mask_vot:
    print('the input is a binary mask')
    selection = handle.region()
    mask = make_full_size(selection, (im.shape[1], im.shape[0]))
    bbox = rect_from_mask(mask)  # [cx,cy,w,h] TODO: use cv.minmaxRect here 
    cx, cy, w, h = bbox
else:
    print('the input is a rect box')
    selection = handle.region()   # selection in ncc_mask
    lx, ly, w, h = selection.x, selection.y, selection.width, selection.height
    cx, cy = lx + w/2, ly + h/2

target_pos = np.array([cx, cy])
target_sz = np.array([w, h])
state = tracker.init(im, target_pos, target_sz, net, mask=mask)


count = 0
while True:
    image_file = handle.frame()
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC
    state = tracker.track(state, im)
    mask = state['mask']
    if mask is None or mask.sum() < 10:
        rect = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        mask = mask_from_rect(rect, (im.shape[1], im.shape[0]))
    handle.report(mask, state['cls_score'])
    count += 1
