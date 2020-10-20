import _init_paths
import os
import cv2
import json
import torch
import random
import argparse
import numpy as np
import models.models as models
from tqdm import tqdm
try:
    from torch2trt import TRTModule
except:
    print('Warning: TensorRT is not successfully imported')
from PIL import Image
from os.path import exists, join, dirname, realpath
from tracker.oceanplus import OceanPlus
from tracker.online import ONLINE
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou
import pdb

def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    parser.add_argument('--arch', dest='arch', default='OceanPlus', choices=['OceanPlus', 'OceanPlusTRT'], help='backbone architecture')
    parser.add_argument('--mms', default='True', type=str, choices=['True', 'False'], help='wether to use MMS')
    parser.add_argument('--resume', default="snapshot/OceanPlusMMS.pth", type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2020', help='dataset test')
    parser.add_argument('--online', action="store_true", help='whether to use online')
    parser.add_argument('--vis', action="store_true", help='visualize tracking results')
    parser.add_argument('--hp', default=None, type=str, help='hyper-parameters')
    parser.add_argument('--debug', default=False, type=str, help='debug or not')
    args = parser.parse_args()

    return args


def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0

    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))


def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)
    mask = rle_to_mask(rle, region_w, region_h)

    return mask

def save_prediction(prediction, palette, save_path, save_name):
    if prediction.ndim > 2:
        img = Image.fromarray(np.uint8(prediction[0, ...]))
    else:
        img = Image.fromarray(np.uint8(prediction))
    img = img.convert('L')
    img.putpalette(palette)
    img = img.convert('P')
    img.save('{}/{}.png'.format(save_path, save_name))


def track(siam_tracker, online_tracker, siam_net, video, args):
    """
    track a single video in VOT2020
    attention: not for benchmark evaluation, just a demo
    TODO: add cyclic initiation
    """

    start_frame, toc = 0, 0
    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if args.online:
            rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # align with training

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            lx, ly, w, h = eval(gt[f][1:])[:4]
            cx = lx + w / 2
            cy = ly + h / 2

            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])

            mask_roi = create_mask_from_string(eval(gt[f][1:]))
            hi, wi, _ = im.shape
            mask_gt = np.zeros((hi, wi))
            mask_gt[ly:ly + h, lx:lx + w] = mask_roi

            state = siam_tracker.init(im, target_pos, target_sz, siam_net, online=args.online, mask=mask_gt, debug=args.debug)  # init siamese tracker

            if args.online:
                online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset, resume=args.resume)

        elif f > start_frame:  # tracking
            if args.online:
                state = online_tracker.track(im, rgb_im, siam_tracker, state)
            else:
                state = siam_tracker.track(state, im, name=image_file)
            mask = state['mask']

            if args.vis:
                COLORS = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
                COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
                mask = COLORS[mask]
                output = ((0.4 * im) + (0.6 * mask)).astype("uint8")
                cv2.imshow("mask", output)
                cv2.waitKey(1)

        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def track_vos(siam_tracker, online_tracker, siam_net, video, args, hp=None):
    re = args.resume.split('/')[-1].split('.')[0]

    if hp is None:
        save_path = join('result', args.dataset, re, video['name'])
    else:
        # re = re+'_thr_{:.2f}_lambdaU_{:.2f}_lambdaS_{:.2f}_iter1_{:.2f}_iter2_{:.2f}'.format(hp['seg_thr'], hp['lambda_u'], hp['lambda_s'], hp['iter1'], hp['iter2'])
        re = re+'_pk_{:.3f}_wi_{:.2f}_lr_{:.2f}'.format(hp['penalty_k'], hp['window_influence'], hp['lr'])
        save_path = join('result', args.dataset, re, video['name'])

    if exists(save_path):
        return

    image_files = video['image_files']
    annos = [Image.open(x) for x in video['anno_files'] if exists(x)]
    palette = annos[0].getpalette()
    annos = [np.array(an) for an in annos]

    if 'anno_init_files' in video:
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]

    mot_enable = args.dataset in ['DAVIS2017', 'YTBVOS']

    if not mot_enable:
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]
    else:
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
        if len(object_ids) != len(annos_init):
            annos_init = annos_init*len(object_ids)
    object_num = len(object_ids)
    toc = 0
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1]))
    for obj_id, o_id in enumerate(object_ids):
        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)

        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            if args.online:
                rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            tic = cv2.getTickCount()
            if f == start_frame:  # init
                mask = annos_init[obj_id] == o_id
                mask = mask.astype(np.uint8)
                x, y, w, h = cv2.boundingRect(mask)
                cx, cy = x + w/2, y + h/2
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                state = siam_tracker.init(im, target_pos, target_sz, siam_net, online=args.online, mask=mask, hp=hp, debug=args.debug)  # init tracker

                if args.online:
                    online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset,
                                        resume=args.resume)
                pred_masks[obj_id, f, :, :] = mask
            elif end_frame >= f > start_frame:  # tracking
                if args.online:
                    state = online_tracker.track(im, rgb_im, siam_tracker, state, name=image_file)
                else:
                    state = siam_tracker.track(state, im, name=image_file)
                mask = state['mask']   # binary
                mask_ori = state['mask_ori']   # probabilistic
            toc += cv2.getTickCount() - tic
            if end_frame >= f >= start_frame:
                if f == start_frame:
                    pred_masks[obj_id, f, :, :] = mask
                else:
                    if args.dataset in ['DAVIS2017', 'YTBVOS']:   # multi-object
                        pred_masks[obj_id, f, :, :] = mask_ori
                    else:
                        pred_masks[obj_id, f, :, :] = mask

            if args.vis:
                COLORS = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
                COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
                mask = COLORS[mask]
                output = ((0.4 * im) + (0.6 * mask)).astype("uint8")
                cv2.imshow("mask", output)
                cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save for evaluation

    if not exists(save_path):
        os.makedirs(save_path)

    if args.dataset == 'DAVIS2016':
        for idx in range(f+1):
            save_name = str(idx).zfill(5)
            save_prediction(pred_masks[:, idx, ...], palette, save_path, save_name)
    elif args.dataset in ['DAVIS2017', 'YTBVOS']:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        for idx in range(f+1):
            if not args.dataset == 'YTBVOS':
                save_name = str(idx).zfill(5)
            else:
                save_name = image_files[idx].split('/')[-1].split('.')[0]

            save_prediction(pred_mask_final[idx, ...], palette, save_path, save_name)
    else:
        raise ValueError('not supported dataset')

    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, (f+1) / toc))


def main():
    print('Warning: this is a demo to test OceanPlus')
    print('Warning: if you want to test it on VOT2020, please use our integration scripts')
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.online = args.online
    info.TRT = 'TRT' in args.arch

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.vis = args.vis
    siam_tracker = OceanPlus(siam_info)

    if args.mms == 'True':
        MMS = True
    else:
        MMS = False
    siam_net = models.__dict__[args.arch](online=args.online, mms=MMS)
    print('===> init Siamese <====')
    siam_net = load_pretrain(siam_net, args.resume)
    siam_net.eval()
    siam_net = siam_net.cuda()

    # if info.TRT:
    #     print('===> load model from TRT <===')
    #     print('===> please ignore the warning information of TRT <===')
    #     trtNet = reloadTRT()
    #     siam_net.tensorrt_init(trtNet)

    if args.online:
        online_tracker = ONLINE(info)
    else:
        online_tracker = None

    print('====> warm up <====')
    for i in tqdm(range(20)):
        siam_net.template(torch.rand(1, 3, 127, 127).cuda(), torch.rand(1, 127, 127).cuda())
        siam_net.track(torch.rand(1, 3, 255, 255).cuda())

    # prepare video
    print('====> load dataset <====')
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    # hyper-parameters in or not
    if args.hp is None:
        hp = None
    elif isinstance(args.hp, str):
        f = open(join('tune', args.hp), 'r')
        hp = json.load(f)
        f.close()
        print('====> tuning hp: {} <===='.format(hp))
    else:
        raise ValueError('not supported hyper-parameters')

    # tracking all videos in benchmark
    for video in video_keys:
        if args.dataset in ['DAVIS2016', 'DAVIS2017', 'YTBVOS']:  # VOS
            track_vos(siam_tracker, online_tracker, siam_net, dataset[video], args, hp)
        else:  # VOTS (i.e. VOT2020)
            if video == 'butterfly':
                track(siam_tracker, online_tracker, siam_net, dataset[video], args)


if __name__ == '__main__':
    main()

