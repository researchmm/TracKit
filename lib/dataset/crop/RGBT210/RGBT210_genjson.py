# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import json
import numpy as np
from os import listdir
from os.path import join

basepath = '/data/share/RGBT210/'
save = dict()


def genjson():
    videos = listdir(basepath)

    for v in videos:
        save[v] = dict()
        save[v]['name'] = v  # video name

        # save img names
        v_in_path = join(basepath, v, 'infrared')
        v_rgb_path = join(basepath, v, 'visible')
        temp1 = listdir(v_in_path)
        temp2 = listdir(v_rgb_path)
        temp1.sort()
        temp2.sort()
        save[v]['infrared_imgs'] = temp1   # infrared file names
        save[v]['visible_imgs'] = temp2    # infrared file names

        # read gt
        v_in_gt_path = join(basepath, v, 'init.txt')
        v_rgb_gt_path = join(basepath, v, 'init.txt')
        v_in_gts = np.loadtxt(v_in_gt_path, delimiter=',')
        v_rgb_gts = np.loadtxt(v_rgb_gt_path, delimiter=',')

        v_in_gts[:, 0:2] = v_in_gts[:, 0:2] - 1    # to python 0 index
        v_rgb_gts[:, 0:2] = v_rgb_gts[:, 0:2] - 1  # to python 0 index

        v_in_init = v_in_gts[0]
        v_rgb_init = v_rgb_gts[0]

        # save int and gt
        save[v]['infrared_init'] = v_in_init.tolist()
        save[v]['visible_init'] = v_rgb_init.tolist()
        save[v]['infrared_gt'] = v_in_gts.tolist()
        save[v]['visible_gt'] = v_rgb_gts.tolist()

    json.dump(save, open('/data/zpzhang/datasets/dataset/RGBT210.json', 'w'), indent=4, sort_keys=True)


if __name__ == '__main__':
    genjson()



