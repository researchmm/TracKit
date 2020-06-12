
import _init_paths
import os
import yaml
import argparse
from os.path import exists
from utils.utils import load_yaml, extract_logs

def parse_args():
    """
    args for onekey.
    """
    parser = argparse.ArgumentParser(description='Train SiamFC with onekey')
    # for train
    parser.add_argument('--cfg', type=str, default='experiments/train/Ocean.yaml', help='yaml configure file name')

    # for

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # train - test - tune information
    info = yaml.load(open(args.cfg, 'r').read())
    info = info['OCEAN']
    trainINFO = info['TRAIN']
    testINFO = info['TEST']
    tuneINFO = info['TUNE']
    dataINFO = info['DATASET']

    # epoch training -- train 50 or more epochs
    if trainINFO['ISTRUE']:
        print('==> train phase')
        print('python ./tracking/train_ocean.py --cfg {0} --gpus {1} --workers {2} 2>&1 | tee logs/ocean_train.log'
                  .format(args.cfg, info['GPUS'], info['WORKERS']))

        if not exists('logs'):
            os.makedirs('logs')

        os.system('python ./tracking/train_ocean.py --cfg {0} --gpus {1} --workers {2} 2>&1 | tee logs/siamrpn_train.log'
                  .format(args.cfg, info['GPUS'], info['WORKERS']))

    # epoch testing -- test 30-50 epochs (or more)
    if testINFO['ISTRUE']:
        print('==> test phase')
        print('mpiexec -n {0} python ./tracking/test_epochs.py --arch {1} --start_epoch {2} --end_epoch {3} --gpu_nums={4} \
                  --threads {0} --dataset {5}  --align {6} 2>&1 | tee logs/ocean_epoch_test.log'
                  .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                          (len(info['GPUS']) + 1) // 2, testINFO['DATA'], trainINFO['ALIGN']))

        if not exists('logs'):
            os.makedirs('logs')

        os.system('mpiexec -n {0} python ./tracking/test_epochs.py --arch {1} --start_epoch {2} --end_epoch {3} --gpu_nums={4} \
                  --threads {0} --dataset {5}  --align {6} 2>&1 | tee logs/ocean_epoch_test.log'
                  .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                          (len(info['GPUS']) + 1) // 2, testINFO['DATA'], trainINFO['ALIGN']))

        # test on vot or otb benchmark
        print('====> use new testing toolkit')
        trackers = os.listdir(os.path.join('./result', testINFO['DATA']))
        trackers = " ".join(trackers)
        if 'VOT' in testINFO['DATA']:
            print('python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset {0} --tracker_result_dir result/{0} --trackers {1}'.format(testINFO['DATA'], trackers))
            os.system('python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset {0} --tracker_result_dir result/{0} --trackers {1} 2>&1 | tee logs/ocean_eval_epochs.log'.format(testINFO['DATA'], trackers))
        else:
            raise ValueError('not supported now, please add new dataset')

    # tuning -- with TPE
    if tuneINFO['ISTRUE']:

        if 'VOT' in testINFO['DATA']:   # for vot real-time and baseline
            resume = extract_logs('logs/ocean_eval_epochs.log', 'VOT')
        else:
            raise ValueError('not supported now')

        print('==> tune phase')
        print('python -u ./tracking/tune_tpe.py --arch {0} --resume {1} --dataset {2} --gpu_nums {3} --align {4}\
                  2>&1 | tee logs/tpe_tune.log'.format(trainINFO['MODEL'], 'snapshot/'+ resume, tuneINFO['DATA'], (len(info['GPUS']) + 1) // 2, trainINFO['ALIGN']))

        if not exists('logs'):
            os.makedirs('logs')
        os.system('python -u ./tracking/tune_tpe.py --arch {0} --resume {1} --dataset {2} --gpu_nums {3} --align {4}\
                  2>&1 | tee logs/tpe_tune.log'.format(trainINFO['MODEL'], 'snapshot/'+ resume, tuneINFO['DATA'], (len(info['GPUS']) + 1) // 2, trainINFO['ALIGN']))


if __name__ == '__main__':
    main()
