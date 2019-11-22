import sys
import json
import os
import glob
from os.path import join, realpath, dirname
import numpy as np

RGBT234 = ['soccer2', 'man45', 'dog11', 'manafterrain', 'woman2', 'nightcar', 'boundaryandfast', 'elecbikeinfrontcar',
            'tricycle1', 'bicyclecity', 'oldman2', 'call', 'manlight', 'baginhand', 'scooter', 'walkingman41', 'children4',
            'woman100', 'walkingwithbag1', 'takeout', 'twoman', 'elecbikewithhat', 'run1', 'twowoman', 'kite4', 'walkingnight',
            'walkingtogether', 'soccer', 'man24', 'car66', 'oldwoman', 'threeman2', 'graycar2', 'man28', 'manonboundary',
            'basketball2', 'playsoccer', 'hotglass', 'guidepost', 'straw', 'people', 'blackwoman', 'luggage', 'man8',
            'manontricycle', 'manwithbag', 'woman', 'threepeople', 'mancross1', 'shoeslight', 'maninblack', 'toy3',
            'maninglass', 'elecbikechange2', 'flower2', 'cycle3', 'elecbikewithlight', 'children2', 'carred', 'womanred',
            'orangeman1', 'elecbike10', 'night2', 'elecbike3', 'woamn46', 'manypeople2', 'tricycletwo', 'rmo', 'woman4',
            'kite2', 'child', 'bikemove1', 'carnotmove', 'womaninblackwithbike', 'man26', 'bike', 'greentruck', 'soccerinhand',
            'nightrun', 'maningreen2', 'afterrain', 'mobile', 'toy1', 'cycle4', 'child3', 'tricyclefaraway', 'jump',
            'yellowcar', 'car4', 'manwithbasketball', 'baketballwaliking', 'cycle2', 'toy4', 'man69', 'man22', 'people3',
            'crouch', 'twoelecbike1', 'car41', 'man23', 'single3', 'manwithbag4', 'manfaraway', 'single1', 'woman6',
            'flower1', 'maninred', 'fog6', 'tree5', 'car3', 'man29', 'tallman', 'manonelecbike', 'children3', 'dog',
            'twoperson', 'run2', 'twoman2', 'twowoman1', 'hotkettle', 'woman48', 'elecbikewithlight1', 'bikeman', 'kettle',
            'womanrun', 'walkingman12', 'man5', 'elecbike', 'greywoman', 'walkingtogetherright', 'carLight', 'caraftertree',
            'whitecar', 'shake', 'supbus2', 'tree3', 'cycle1', 'car37', 'floor-1', 'man4', 'bus6', 'aftertree', 'redmanchange',
            'tree2', 'face1', 'walkingwithbag2', 'nightthreepeople', 'inglassandmobile', 'woman89', 'crossroad', 'diamond',
            'woman3', 'womanfaraway', 'redcar2', 'manypeople', 'bluebike', 'blueCar', 'mancross', 'together', 'man9',
            'whitebag', 'manwithumbrella', 'walkingman1', 'threeman', 'womanleft', 'child1', 'baby', 'whiteman1', 'glass',
            'manoccpart', 'whitecar3', 'push', 'biketwo', 'man55', 'child4', 'redbag', 'walkingman', 'mancrossandup',
            'walkingman20', 'manup', 'rainingwaliking', 'womanpink', 'oldman', 'tricycle9', 'womancross', 'tricycle6',
            'man3', 'notmove', 'manwithluggage', 'cycle5', 'walking41', 'man68', 'car20', 'woman1', 'woamnwithbike',
            'glass2', 'whitecar4', 'manout2', 'man2', 'dog10', 'walking40', 'whitecarafterrain', 'womanwithbag6',
            'people1', 'walkingwoman', 'elecbike2', 'carnotfar', 'dog1', 'raningcar', 'woman99', 'trees', 'balancebike',
            'threewoman2', 'run', 'stroller', 'man88', 'tricycle', 'redcar', 'fog', 'twoelecbike', 'walkingmantiny',
            'supbus', 'walkingtogether1', 'mandrivecar', 'manypeople1', 'green', 'greyman', 'car10', 'twoman1',
            'tricycle2', 'whitesuv', 'man7', 'woman96', 'car']


def getmax(iou1, iou2):
    """
    compare infrid and rgb ious (each frame by frame), choose the larger one
    """
    iou = [over1 if over1 > iou2[idx] else iou2[idx] for idx, over1 in enumerate(iou1)]
    return np.array(iou)


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def compute_success_overlap(in_gt_bb, rgb_gt_bb, bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(in_gt_bb)
    success = np.zeros(len(thresholds_overlap))

    in_iou = overlap_ratio(in_gt_bb, bb)      # iou of infrid data
    rgb_iou = overlap_ratio(rgb_gt_bb, bb)   # iou of rgb data
    iou = getmax(in_iou, rgb_iou)

    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success


def get_result_bb(arch, seq):
    result_path = join(arch, arch.split('/')[-1] + '_' + seq + '.txt')
    try:
        temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
    except:
        temp = np.loadtxt(result_path, delimiter=' ').astype(np.float)
    return np.array(temp)


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def eval_auc(result_path='./test/', tracker_reg='S*', start=0, end=1e6):
    dataset = 'RGBT234'
    list_path = os.path.join(realpath(dirname(__file__)), '../../', 'dataset', dataset + '.json')
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())  # dict to list for py3

    trackers = glob.glob(join(result_path, dataset, tracker_reg))
    trackers = trackers[start:min(end, len(trackers))]

    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    # thresholds_error = np.arange(0, 51, 1)

    success_overlap = np.zeros((n_seq, len(trackers), len(thresholds_overlap)))
    # success_error = np.zeros((n_seq, len(trackers), len(thresholds_error)))
    for i in range(n_seq):
        seq = seqs[i]
        in_gt_rect = np.array(annos[seq]['infrared_gt']).astype(np.float)  # 0-index
        rgb_gt_rect = np.array(annos[seq]['visible_gt']).astype(np.float)  # 0-index
        in_gt_rect[:, 0:2] = in_gt_rect[:, 0:2] + 1       # 1-index
        rgb_gt_rect[:, 0:2] = rgb_gt_rect[:, 0:2] + 1     # 1-index

        for j in range(len(trackers)):
            tracker = trackers[j]
            print('{:d} processing:{} tracker: {}'.format(i, seq, tracker))
            bb = get_result_bb(tracker, seq)
            success_overlap[i][j] = compute_success_overlap(in_gt_rect, rgb_gt_rect, bb)
            # success_error[i][j] = compute_success_error(gt_center, center)

    print('Success Overlap')

    max_auc = 0.
    max_name = ''
    for i in range(len(trackers)):
        auc = success_overlap[:, i, :].mean()
        if auc > max_auc:
            max_auc = auc
            max_name = trackers[i]
        print('%s(%.4f)' % (trackers[i], auc))

    print('\n%s Best: %s(%.4f)' % (dataset, max_name, max_auc))


def eval_rgbt234_tune(result_path):
    dataset = 'RGBT234'
    list_path = os.path.join(realpath(dirname(__file__)), '../../', 'dataset', dataset + '.json')
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())  # dict to list for py3
    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success_overlap = np.zeros((n_seq, 1, len(thresholds_overlap)))

    for i in range(n_seq):
        seq = seqs[i]
        in_gt_rect = np.array(annos[seq]['infrared_gt']).astype(np.float)  #
        rgb_gt_rect = np.array(annos[seq]['visible_gt']).astype(np.float)  #
        in_gt_rect[:, 0:2] = in_gt_rect[:, 0:2] + 1
        rgb_gt_rect[:, 0:2] = rgb_gt_rect[:, 0:2] + 1

        bb = get_result_bb(result_path, seq)
        success_overlap[i][0] = compute_success_overlap(in_gt_rect, rgb_gt_rect, bb)

    auc = success_overlap[:, 0, :].mean()
    return auc


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('python ./lib/core/eval_rgbt234.py ./result SiamFC* 0 1')
        exit()
    result_path = sys.argv[1]
    tracker_reg = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    eval_auc(result_path, tracker_reg, start, end)
