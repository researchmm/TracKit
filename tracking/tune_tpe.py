from __future__ import absolute_import
import _init_paths
import os
import argparse
import numpy as np

import models.models as models
from utils.utils import load_pretrain
from test_ocean import auc_otb, eao_vot
from tracker.ocean import Ocean
from easydict import EasyDict as edict

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch
from hyperopt import hp
from pprint import pprint

parser = argparse.ArgumentParser(description='parameters for SiamFC tracker')
parser.add_argument('--arch', dest='arch', default='Ocean',
                    help='architecture of model')
parser.add_argument('--resume', default='', type=str, required=True,
                    help='resumed model')
parser.add_argument('--gpu_nums', default=4, type=int, help='gpu numbers')
parser.add_argument('--dataset', default='OTB2013', type=str, help='dataset')
parser.add_argument('--align', default='True', type=str, help='align')

args = parser.parse_args()

print('==> However TPE is slower than GENE')

# prepare tracker 
info = edict()
info.arch = args.arch
info.dataset = args.dataset
info.epoch_test = False

# create model
info.align = True if 'VOT' in args.dataset and args.align=='True' else False
if 'Ocean' in args.arch:
    model = models.__dict__[args.arch](align=info.align)
    tracker = Ocean(info)
else:
    raise ValueError('not supported other model now')

model = load_pretrain(model, args.resume)
model.eval()
model = model.cuda()
print('pretrained model has been loaded')


# fitness function
def fitness(config, reporter):
    if 'Ocean' in args.arch:
        penalty_k = config["penalty_k"]
        scale_lr = config["scale_lr"]
        window_influence = config["window_influence"]
        small_sz = config["small_sz"]
        big_sz = config["big_sz"]
        ratio = config["ratio"]

        model_config = dict()
        model_config['benchmark'] = args.dataset
        model_config['arch'] = args.arch
        model_config['resume'] = args.resume
        model_config['hp'] = dict()
        model_config['hp']['penalty_k'] = penalty_k
        model_config['hp']['window_influence'] = window_influence
        model_config['hp']['lr'] = scale_lr
        model_config['hp']['small_sz'] = small_sz
        model_config['hp']['big_sz'] = big_sz
        model_config['hp']['ratio'] = ratio
    else:
        raise ValueError('not supported model')


    # VOT and Ocean
    if args.dataset.startswith('VOT'):
        eao = eao_vot(tracker, model, model_config)
        print("penalty_k: {0}, scale_lr: {1}, window_influence: {2}, small_sz: {3}, big_sz: {4}, ratio: {6}, eao: {5}".format(penalty_k, scale_lr, window_influence, small_sz, big_sz, eao, ratio))
        reporter(EAO=eao)

    # OTB and Ocean
    if args.dataset.startswith('OTB'):
        auc = auc_otb(tracker, model, model_config)
        print("penalty_k: {0}, scale_lr: {1}, window_influence: {2}, small_sz: {3}, big_sz: {4}, ratio: {6}, eao: {5}".format(penalty_k, scale_lr, window_influence, small_sz, big_sz, auc.item(), ratio))
        reporter(AUC=auc)



if __name__ == "__main__":
    # the resources you computer have, object_store_memory is shm
    ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8, redirect_output=True, object_store_memory=50000000000)
    tune.register_trainable("fitness", fitness)

    if 'Ocean' in args.arch:
        params = {
                "penalty_k": hp.quniform('penalty_k', 0.001, 0.6, 0.001),
                "scale_lr": hp.quniform('scale_lr', 0.1, 0.8, 0.001),
                "window_influence": hp.quniform('window_influence', 0.05, 0.65, 0.001),
                "small_sz": hp.choice("small_sz", [255, 271]),
                "big_sz": hp.choice("big_sz", [271, 287]),
                "ratio": hp.quniform('ratio', 0.5, 1, 0.01),
                }
    if 'VOT' not in args.dataset:
        params['ratio'] = hp.choice("ratio", [1]) 
 
    print('tuning range: ')
    pprint(params)    

    tune_spec = {
        "zp_tune": {
            "run": "fitness",
            "trial_resources": {
                "cpu": 1,  # single task cpu num
                "gpu": 0.5,  # single task gpu num
            },
            "num_samples": 10000,  # sample hyperparameters times
            "local_dir": './TPE_results'
        }
    }

    # stop condition for VOT and OTB
    if args.dataset.startswith('VOT'):
        stop = {
            "EAO": 0.6,  # if EAO >= 0.6, this procedures will stop
            # "timesteps_total": 100, # iteration times
        }
        tune_spec['zp_tune']['stop'] = stop

        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            reward_attr="EAO",
            max_t=400,
            grace_period=20
        )
        algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*2 + 1, reward_attr="EAO") # max_concurrent: the max running task

    elif args.dataset.startswith('OTB')  or args.dataset.startswith('VIS') or args.dataset.startswith('GOT10K'):
        stop = {
            # "timesteps_total": 100, # iteration times
            "AUC": 0.80
        }
        tune_spec['zp_tune']['stop'] = stop
        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            reward_attr="AUC",
            max_t=400,
            grace_period=20
        )
        algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*2 + 1, reward_attr="AUC")  #
    else:
        raise ValueError("not support other dataset now")

    tune.run_experiments(tune_spec, search_alg=algo, scheduler=scheduler)



