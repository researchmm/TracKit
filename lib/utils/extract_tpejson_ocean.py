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
parser.add_argument('--dataset', default='VOT2019', help='test dataset')
parser.add_argument('--save_path', default='logs', help='log file save path')


def collect_results(args):
    dirs = os.listdir(args.path)
    print('[*] ===== total {} files in TPE dir'.format(len(dirs)))

    count = 0
    penalty_k = []
    scale_lr = []
    wi = []
    big_sz = []
    small_sz = []
    ratio = []
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
            # pdb.set_trace()
            eao.append(js['EAO'])
            temp = js['config']
            scale_lr.append(temp["scale_lr"])
            wi.append(temp["window_influence"])
            penalty_k.append(temp["penalty_k"])
            ratio.append(temp["ratio"])
            small_sz.append(temp["small_sz"])
            big_sz.append(temp["big_sz"])
 
            
    # find max
    print('{} params group  have been tested'.format(count))
    eao = np.array(eao)
    max_idx = np.argmax(eao)
    max_eao = eao[max_idx]
    print('penalty_k: {:.4f}, scale_lr: {:.4f}, wi: {:.4f}, ratio: {:.4f}, small_sz: {}, big_sz: {:.4f}, eao: {}'.format(penalty_k[max_idx], scale_lr[max_idx], wi[max_idx], ratio[max_idx], small_sz[max_idx], big_sz[max_idx], max_eao))


if __name__ == '__main__':
    args = parser.parse_args()
    collect_results(args)