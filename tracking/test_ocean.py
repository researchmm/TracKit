# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------
import _init_paths
import os
import cv2
import torch
import random
import argparse
import numpy as np

try:
    from torch2trt import TRTModule
except:
    print('Warning: TensorRT is not successfully imported')

import models.models as models

from os.path import exists, join, dirname, realpath
from tracker.ocean import Ocean
from tracker.online import ONLINE
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

from eval_toolkit.pysot.datasets import VOTDataset
from eval_toolkit.pysot.evaluation import EAOBenchmark
from lib.core.eval_otb import eval_auc_tune
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Test Ocean')
    parser.add_argument('--arch', dest='arch', default='OceanTRT', help='backbone architecture')
    parser.add_argument('--resume', default="snapshot/OceanV.pth", type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--align', default='True', type=str, help='alignment module flag') # bool 
    parser.add_argument('--online', default=False, type=bool, help='online flag')
    parser.add_argument('--video', default=None, type=str, help='test a video in benchmark')
    args = parser.parse_args()

    return args

def reloadTRT():
    absPath = os.path.abspath(os.path.dirname(__file__))
    t_bk_path = join(absPath, '../', 'snapshot', 't_backbone.pth')
    s_bk_siam255_path = join(absPath, '../', 'snapshot', 's_backbone_siam255.pth')
    s_bk_siam287_path = join(absPath, '../', 'snapshot', 's_backbone_siam287.pth')
    s_bk_online_path = join(absPath, '../', 'snapshot', 's_backbone_online.pth')
    t_neck_path = join(absPath, '../', 'snapshot', 't_neck.pth')
    s_neck255_path = join(absPath, '../', 'snapshot', 's_neck255.pth')
    s_neck287_path = join(absPath, '../', 'snapshot', 's_neck287.pth')
    multiDiCorr255_path = join(absPath, '../', 'snapshot', 'multiDiCorr255.pth')
    multiDiCorr287_path = join(absPath, '../', 'snapshot', 'multiDiCorr287.pth')
    boxtower255_path = join(absPath, '../', 'snapshot', 'boxtower255.pth')
    boxtower287_path = join(absPath, '../', 'snapshot', 'boxtower287.pth')

    t_bk = TRTModule()
    s_bk_siam255 = TRTModule()
    s_bk_siam287 = TRTModule()
    s_bk_online = TRTModule()
    t_neck = TRTModule()
    s_neck255 = TRTModule()
    s_neck287 = TRTModule()
    multiDiCorr255 = TRTModule()
    multiDiCorr287 = TRTModule()
    boxtower255 = TRTModule()
    boxtower287 = TRTModule()

    t_bk.load_state_dict(torch.load(t_bk_path))
    s_bk_siam255.load_state_dict(torch.load(s_bk_siam255_path))
    s_bk_siam287.load_state_dict(torch.load(s_bk_siam287_path))
    s_bk_online.load_state_dict(torch.load(s_bk_online_path))
    t_neck.load_state_dict(torch.load(t_neck_path))
    s_neck255.load_state_dict(torch.load(s_neck255_path))
    s_neck287.load_state_dict(torch.load(s_neck287_path))
    multiDiCorr255.load_state_dict(torch.load(multiDiCorr255_path))
    multiDiCorr287.load_state_dict(torch.load(multiDiCorr287_path))
    boxtower255.load_state_dict(torch.load(boxtower255_path))
    boxtower287.load_state_dict(torch.load(boxtower287_path))

    return [t_bk, s_bk_siam255, s_bk_siam287, s_bk_online, t_neck, s_neck255, s_neck287, multiDiCorr255, multiDiCorr287, boxtower255, boxtower287]


def track(siam_tracker, online_tracker, siam_net, video, args):
    start_frame, toc = 0, 0

    # save result to evaluate
    if args.epoch_test:
        suffix = args.resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('result', args.dataset, args.arch + suffix)
    else:
        tracker_path = os.path.join('result', args.dataset, args.arch)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return  # for mult-gputesting

    regions = []
    lost = 0

    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # align with training

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])

            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])

            state = siam_tracker.init(im, target_pos, target_sz, siam_net)  # init tracker

            if args.online:
                online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset, resume=args.resume)

            # location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            if args.online:
                state = online_tracker.track(im, rgb_im, siam_tracker, state)
            else:
                state = siam_tracker.track(state, im)

            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
                lost += 1
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        elif 'OTB' in args.dataset or 'LASOT' in args.dataset:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        elif 'VISDRONE' in args.dataset or 'GOT10K' in args.dataset:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video['name'], toc, f / toc, lost))


def main():
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.TRT = 'TRT' in args.arch
    info.epoch_test = args.epoch_test

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.online = args.online
    siam_info.epoch_test = args.epoch_test
    siam_info.TRT = 'TRT' in args.arch
    if args.online:
        siam_info.align = False
    else:
        siam_info.align = True if 'VOT' in args.dataset and args.align=='True' else False

    if siam_info.TRT:
        siam_info.align = False

    siam_tracker = Ocean(siam_info)
    siam_net = models.__dict__[args.arch](align=siam_info.align, online=args.online)
    print(siam_net)
    print('===> init Siamese <====')

    if not siam_info.TRT:
        siam_net = load_pretrain(siam_net, args.resume)
    else:
        print("tensorrt toy model: not loading checkpoint")
    siam_net.eval()
    siam_net = siam_net.cuda()

    if siam_info.TRT:
        print('===> load model from TRT <===')
        print('===> please ignore the warning information of TRT <===')
        print('===> We only provide a toy demo for TensorRT. There are some operations are not supported well.<===')
        print('===> If you wang to test on benchmark, please us Pytorch version. <===')
        print('===> The tensorrt code will be contingously optimized (with the updating of official TensorRT.)<===')
        trtNet = reloadTRT()
        siam_net.tensorrt_init(trtNet)

    if args.online:
        online_tracker = ONLINE(info)
    else:
        online_tracker = None

    print('====> warm up <====')
    for i in tqdm(range(100)):
        siam_net.template(torch.rand(1, 3, 127, 127).cuda())
        siam_net.track(torch.rand(1, 3, 255, 255).cuda())

    # prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()
    
    if args.video is not None:
        track(siam_tracker, online_tracker, siam_net, dataset[args.video], args)
    else:
        for video in video_keys:
            track(siam_tracker, online_tracker, siam_net, dataset[video], args)


# -----------------------------------------------
# The next few functions are utilized for tuning
# -----------------------------------------------
def track_tune(tracker, net, video, config):
    arch = config['arch']
    benchmark_name = config['benchmark']
    resume = config['resume']
    hp = config['hp']  # scale_step, scale_penalty, scale_lr, window_influence

    tracker_path = join('test', (benchmark_name + resume.split('/')[-1].split('.')[0] +
                                     '_small_size_{:.4f}'.format(hp['small_sz']) +
                                     '_big_size_{:.4f}'.format(hp['big_sz']) +
                                     '_ratio_{:.4f}'.format(hp['ratio']) +
                                     '_penalty_k_{:.4f}'.format(hp['penalty_k']) +
                                     '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                     '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .
    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in benchmark_name:
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = join(video_path, video['name'] + '_001.txt')
    elif 'GOT10K' in benchmark_name:
        re_video_path = os.path.join(tracker_path, video['name'])
        if not exists(re_video_path): os.makedirs(re_video_path)
        result_path = os.path.join(re_video_path, '{:s}.txt'.format(video['name']))
    else:
        result_path = join(tracker_path, '{:s}.txt'.format(video['name']))

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        if benchmark_name.startswith('OTB'):
            return tracker_path
        elif benchmark_name.startswith('VOT') or benchmark_name.startswith('GOT10K'):
            return 0
        else:
            print('benchmark not supported now')
            return

    start_frame, lost_times, toc = 0, 0, 0

    regions = []  # result and states[1 init / 2 lost / 0 skip]

    # for rgbt splited test

    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = tracker.init(im, target_pos, target_sz, net, hp=hp)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in benchmark_name else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append([float(2)])
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append([float(0)])

    # save results for OTB
    if 'OTB' in benchmark_name or 'LASOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VISDRONE' in benchmark_name  or 'GOT10K' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    if 'OTB' in benchmark_name or 'VIS' in benchmark_name or 'VOT' in benchmark_name or 'GOT10K' in benchmark_name:
        return tracker_path
    else:
        print('benchmark not supported now')


def auc_otb(tracker, net, config):
    """
    get AUC for OTB benchmark
    """
    dataset = load_dataset(config['benchmark'])
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    auc = eval_auc_tune(result_path, config['benchmark'])

    return auc

def eao_vot(tracker, net, config):
    dataset = load_dataset(config['benchmark'])
    video_keys = sorted(list(dataset.keys()).copy())

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    re_path = result_path.split('/')[0]
    tracker = result_path.split('/')[-1]

    # debug
    print('======> debug: results_path')
    print(result_path)
    print(os.system("ls"))
    print(join(realpath(dirname(__file__)), '../dataset'))

    # give abs path to json path
    data_path = join(realpath(dirname(__file__)), '../dataset')
    dataset = VOTDataset(config['benchmark'], data_path)

    dataset.set_tracker(re_path, tracker)
    benchmark = EAOBenchmark(dataset)
    eao = benchmark.eval(tracker)
    eao = eao[tracker]['all']

    return eao


if __name__ == '__main__':
    main()

