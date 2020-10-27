# -*- coding:utf-8 -*-
# ! ./usr/bin/env python

import os
import json
import shutil
import argparse
import numpy as np
import pdb


parser = argparse.ArgumentParser(description='Analysis siamfc tune results')
parser.add_argument('--path', default='./TPE_results/zp_tune', help='tune result path')
parser.add_argument('--dataset', default='VOT2018', help='test dataset')
parser.add_argument('--save_path', default='logs', help='log file save path')


def collect_results(args):
    dirs = os.listdir(args.path)
    print('[*] ===== total {} files in TPE dir'.format(len(dirs)))

    count = 0
    scale_penalty = []
    scale_lr = []
    wi = []
    scale_step = []
    eao = []
    count = 0 # total numbers

    for d in dirs:
        param_path = os.path.join(args.path, d)
        json_path = os.path.join(param_path, 'result.json')

        if not os.path.exists(json_path):
            continue

        # pdb.set_trace()
        try:
            js = json.load(open(json_path, 'r'))
        except:
            continue
 
        if not "EAO" in list(js.keys()):
            continue
        else:
            count += 1
            eao.append(js['EAO'])
            temp = js['config']
            scale_lr.append(temp["scale_lr"])
            wi.append(temp["w_influence"])
            scale_step.append(temp["scale_step"])
            scale_penalty.append(temp["scale_penalty"])
 
            
    # find max
    print('{} params group  have been tested'.format(count))
    eao = np.array(eao)
    max_idx = np.argmax(eao)
    max_eao = eao[max_idx]
    print('scale_penalty: {:.4f}, scale_lr: {:.4f}, wi: {:.4f}, scale_step: {}, eao: {}'.format(scale_penalty[max_idx], scale_lr[max_idx], wi[max_idx], scale_step[max_idx], max_eao))


if __name__ == '__main__':
    args = parser.parse_args()
    collect_results(args)