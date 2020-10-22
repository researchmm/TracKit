# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# multi-gpu test for epochs
# ------------------------------------------------------------------------------

import os
import time
import argparse
import numpy as np
from os import listdir
from os.path import join, exists
from concurrent import futures

parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--dataset', default='DAVIS2016', type=str, help='benchmarks')
parser.add_argument('--num_threads', default=16, type=int, help='number of threads')
parser.add_argument('--datapath', default='dataset/DAVIS', type=str, help='benchmarks')
args = parser.parse_args()


def eval_davis(epoch):
    year = args.dataset[5:]
    full_path = join('result', args.dataset, epoch)
    os.system('python lib/eval_toolkit/davis/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path {0} --davis_path {1} --year {2}'.format(full_path, args.datapath, year))


def extract_davis(epochs):
    # J&F-Mean,J-Mean,J-Recall,J-Decay,F-Mean,F-Recall,F-Decay
    results = dict()
    print('\t \tJ&F-Mean,J-Mean,J-Recall,J-Decay,F-Mean,F-Recall,F-Decay')

    JFm = []
    Jm = []
    Jr = []
    Jd = []
    Fm = []
    Fr = []
    Fd = []
    
    for e in epochs:
        results[e] = dict()
        full_path = join('result', args.dataset, e, 'global_results-val.csv')
        record = open(full_path, 'r').readlines()
        record = eval(record[1])
        print('{} {} {} {} {} {} {} {}'.format(e, record[0], record[1], record[2], record[3], record[4], record[5], record[6]))

        JFm.append(record[0])
        Jm.append(record[1])
        Jr.append(record[2])
        Jd.append(record[3])
        Fm.append(record[4])
        Fr.append(record[5])
        Fd.append(record[6])
    print('=========> sort with J&F: <===========')
    argidx = np.argmax(np.array(JFm))
    print('{} {} {} {} {} {} {} {}'.format(epochs[argidx], JFm[argidx], Jm[argidx], Jr[argidx], Jd[argidx], Fm[argidx], Fr[argidx], Fd[argidx]))
    print('=========> sort with Jm: <===========')
    argidx = np.argmax(np.array(Jm))
    print('{} {} {} {} {} {} {} {}'.format(epochs[argidx], JFm[argidx], Jm[argidx], Jr[argidx], Jd[argidx], Fm[argidx], Fr[argidx], Fd[argidx]))


base_path = join('result', args.dataset)
epochs = listdir(base_path)
print('total {} epochs'.format(len(epochs)))

# multi-process evaluation
if args.dataset in ['DAVIS2016', 'DAVIS2017']:
    with futures.ProcessPoolExecutor(max_workers=args.num_threads) as executor:
        fs = [executor.submit(eval_davis, e) for e in epochs]
    print('done')
    extract_davis(epochs)
else:
    raise ValueError('not supported data')
