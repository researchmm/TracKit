# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import cv2
import json
import glob
import numpy as np
from os.path import join
from os import listdir

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str, default='/data/share/GOT10K', help='your vid data dir')
args = parser.parse_args()

got10k_base_path = args.dir
sub_sets = sorted({'train', 'val'})

got10k = []
for sub_set in sub_sets:
    sub_set_base_path = join(got10k_base_path, sub_set)
    videos = sorted(listdir(sub_set_base_path))
    s = []
    for vi, video in enumerate(videos):
        print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
        v = dict()
        v['base_path'] = join(sub_set, video)
        v['frame'] = []
        video_base_path = join(sub_set_base_path, video)
        gts_path = join(video_base_path, 'groundtruth.txt')
        # gts_file = open(gts_path, 'r')
        # gts = gts_file.readlines()
        gts = np.loadtxt(open(gts_path, "rb"), delimiter=',')

        # get image size
        im_path = join(video_base_path, '00000001.jpg')
        im = cv2.imread(im_path)
        size = im.shape  # height, width
        frame_sz = [size[1], size[0]]  # width,height

        # get all im name
        jpgs = sorted(glob.glob(join(video_base_path, '*.jpg')))

        f = dict()
        for idx, img_path in enumerate(jpgs):
            f['frame_sz'] = frame_sz
            f['img_path'] = img_path.split('/')[-1]

            gt = gts[idx]
            bbox = [int(g) for g in gt]   # (x,y,w,h)
            f['bbox'] = bbox
            v['frame'].append(f.copy())
        s.append(v)
    got10k.append(s)
print('save json (raw got10k info), please wait 1 min~')
json.dump(got10k, open('got10k.json', 'w'), indent=4, sort_keys=True)
print('got10k.json has been saved in ./')
