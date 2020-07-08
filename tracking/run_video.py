# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Email: zhangzhipeng2017@ia.ac.cn
# Detail: test on a specific video (provide init bbox [optional] and video file)
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
from tqdm import tqdm


def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    parser.add_argument('--arch', default='Ocean', type=str, help='backbone architecture')
    parser.add_argument('--resume', default='snapshot/OceanV19on.pth', type=str, help='pretrained model')
    parser.add_argument('--video', default='./dataset/soccer1.mp4', type=str, help='video file path')
    parser.add_argument('--online', default=True, type=bool, help='use online or offline model')
    parser.add_argument('--save', default=True, type=bool, help='save pictures')
    parser.add_argument('--init_bbox', default=None, help='bbox in the first frame None or [lx, ly, w, h]')
    args = parser.parse_args()

    return args


def track_video(siam_tracker, online_tracker, siam_net, video_path, init_box=None, args=None):

    assert os.path.isfile(video_path), "please provide a valid video file"

    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    save_path = os.path.join('vis', video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)
    display_name = 'Video: {}'.format(video_path.split('/')[-1])
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 960, 720)
    success, frame = cap.read()
    cv2.imshow(display_name, frame)

    if success is not True:
        print("Read failed.")
        exit(-1)

    # init
    count = 0

    if init_box is not None:
        lx, ly, w, h = init_box
        target_pos = np.array([lx + w/2, ly + h/2])
        target_sz = np.array([w, h])

        state = siam_tracker.init(frame, target_pos, target_sz, siam_net)  # init tracker
        rgb_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if args.online:
            online_tracker.init(frame, rgb_im, siam_net, target_pos, target_sz, True, dataname='VOT2019', resume=args.resume)

    else:
        while True:

            frame_disp = frame.copy()

            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       1, (0, 0, 255), 1)

            lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])

            state = siam_tracker.init(frame_disp, target_pos, target_sz, siam_net)  # init tracker
            rgb_im = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)

            if args.online:
                online_tracker.init(frame_disp, rgb_im, siam_net, target_pos, target_sz, True, dataname='VOT2019', resume=args.resume)

            break

    while True:
        ret, frame = cap.read()

        if frame is None:
            return

        frame_disp = frame.copy()
        rgb_im = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)

        # Draw box
        if args.online:
            state = online_tracker.track(frame_disp, rgb_im, siam_tracker, state)
        else:
            state = siam_tracker.track(state, frame_disp)

        location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(location[1] + location[3])

        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 5)

        font_color = (0, 0, 0)
        cv2.putText(frame_disp, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)
        cv2.putText(frame_disp, 'Press r to reset', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)
        cv2.putText(frame_disp, 'Press q to quit', (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)

        # Display the resulting frame
        cv2.imshow(display_name, frame_disp)

        if args.save:
            save_name = os.path.join(save_path, '{:04d}.jpg'.format(count))
            cv2.imwrite(save_name, frame_disp)
            count += 1

        key = cv2.waitKey(1)
        # key = None
        if key == ord('q'):
            break
        elif key == ord('r'):
            ret, frame = cap.read()
            frame_disp = frame.copy()

            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       1.5,
                       (0, 0, 0), 1)

            cv2.imshow(display_name, frame_disp)
            lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])

            state = siam_tracker.init(frame_disp, target_pos, target_sz, siam_net)  # init tracker
            rgb_im = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)

            if args.online:
                online_tracker.init(frame_disp, rgb_im, siam_net, target_pos, target_sz, True, dataname='VOT2019', resume=args.resume)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # prepare model (SiamRPN or SiamFC)

    # prepare tracker
    info = edict()
    info.arch = args.arch
    info.dataset = 'VOT2019'
    info.TRT = 'TRT' in args.arch
    info.epoch_test = False

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = 'VOT2019'
    siam_info.online = args.online
    siam_info.epoch_test = False
    siam_info.TRT = 'TRT' in args.arch

    siam_info.align = False

    if siam_info.TRT:
        siam_info.align = False

    siam_tracker = Ocean(siam_info)
    siam_net = models.__dict__[args.arch](align=siam_info.align, online=args.online)
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

    print('[*] ======= Track video with {} ======='.format(args.arch))

    # check init box is list or not
    if not isinstance(args.init_bbox, list) and args.init_bbox is not None:
        args.init_bbox = list(eval(args.init_bbox))
    else:
        args.init_bbox = None
        print('===> please draw a box with your mouse <====')

    track_video(siam_tracker, online_tracker, siam_net, args.video, init_box=args.init_bbox, args=args)

if __name__ == '__main__':
    main()
