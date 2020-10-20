import os
import json
import glob
import torch
import logging
import time
import math
import torch
import yaml
import cv2
import random
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from collections import namedtuple
from shapely.geometry import Polygon, box
from os.path import join, realpath, dirname, exists
# from utils.visdom import Visdom
from _collections import OrderedDict
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# ---------------------------------
# vis
# ---------------------------------
def visdom_draw_tracking(visdom, image, box, segmentation=None):
    if isinstance(box, OrderedDict):
        box = [v for k, v in box.items()]
    else:
        box = (box,)
    if segmentation is None:
        visdom.register((image, *box), 'Tracking', 1, 'Tracking')
    else:
        visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')

def _visdom_ui_handler(self, data):
    pause_mode = False
    if data['event_type'] == 'KeyPress':
        if data['key'] == ' ':
            pause_mode = not pause_mode

        elif data['key'] == 'ArrowRight' and pause_mode:
            self.step = True

def _init_visdom(visdom_info, debug=False):
    visdom_info = {} if visdom_info is None else visdom_info
    pause_mode = False
    step = False

    visdom = Visdom(debug, {'handler': _visdom_ui_handler, 'win_id': 'Tracking'},
                         visdom_info=visdom_info)

    # Show help
    help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                'block list.'
    visdom.register(help_text, 'text', 1, 'Help')

    return visdom


# ---------------------------------
# Functions for FC tracking tools
# ---------------------------------
def load_yaml(path, subset=True):
    file = open(path, 'r')
    yaml_obj = yaml.load(file.read(), Loader=yaml.FullLoader)

    if subset:
        hp = yaml_obj['TEST']
    else:
        hp = yaml_obj

    return hp


def to_torch(ndarray):
    return torch.from_numpy(ndarray)


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img



def get_subwindow_tracking_mask(im, pos, model_sz, original_sz, out_mode='torch'):
    """
    SiamFC type cropping
    """
    crop_info = dict()

    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad), np.uint8)
        # for return mask
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad:top_pad + r, left_pad:left_pad + c] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c] = 0
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c] = 0
        if left_pad:
            te_im[:, 0:left_pad] = 0
        if right_pad:
            te_im[:, c + left_pad:] = 0
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]
    else:
        tete_im = np.zeros(im.shape[0:2])
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    crop_info['crop_cords'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info['empty_mask'] = tete_im
    crop_info['pad_info'] = [top_pad, left_pad, r, c]

    if out_mode == "torch":
        return im_to_torch(im_patch.copy()), crop_info
    else:
        return im_patch, crop_info


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    SiamFC type cropping
    """
    crop_info = dict()

    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        # for return mask
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        tete_im = np.zeros(im.shape[0:2])
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    crop_info['crop_cords'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info['empty_mask'] = tete_im
    crop_info['pad_info'] = [top_pad, left_pad, r, c]

    if out_mode == "torch":
        return im_to_torch(im_patch.copy()), crop_info
    else:
        return im_patch, crop_info


def make_scale_pyramid(im, pos, in_side_scaled, out_side, avg_chans):
    """
    SiamFC 3/5 scale imputs
    """
    in_side_scaled = [round(x) for x in in_side_scaled]
    num_scale = len(in_side_scaled)
    pyramid = torch.zeros(num_scale, 3, out_side, out_side)
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side

    search_side = round(beta * max_target_side)
    search_region, _ = get_subwindow_tracking(im, pos, int(search_side), int(max_target_side), avg_chans, out_mode='np')

    for s, temp in enumerate(in_side_scaled):
        target_side = round(beta * temp)
        temp, _ = get_subwindow_tracking(search_region, (1 + search_side) / 2, out_side, target_side, avg_chans)
        pyramid[s, :] = temp
    return pyramid

# ---------------------------------
# Functions for FC tracking tools
# ---------------------------------
def python2round(f):
    """
    use python2 round function in python3
    """
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)


def generate_anchor(total_stride, scales, ratios, score_size):
    """
    slight different with released SiamRPN-VOT18
    prefer original size without flatten
    """
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    score_size = int(score_size)
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))

    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])

    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)

    anchor = np.reshape(anchor, (5, score_size, score_size, 4))   # this order is right  [5, 17, 17, 4]
    anchor = np.transpose(anchor, (3, 0, 1, 2))    # [4,5,17,17]

    return anchor   # [4, 5, 17, 17]



class ImageNormalizer(object):

    def __init__(self, mean, std, in_type='opencv', out_type='pil'):
        """
        Normalize input tensor by substracting mean value & scale std value.
        """
        self.mean = mean
        self.std = std

        assert in_type in ("opencv", "pil"), "Type must be 'opencv' or 'pil'"
        assert out_type in ("opencv", "pil"), "Type must be 'opencv' or 'pil'"

        if in_type == out_type:
            self.order_trans = False
            self.scale_factor = 1.0
        elif in_type == 'opencv' and out_type == 'pil':
            self.order_trans = True
            self.div_factor = 255.0
        elif in_type == 'pil' and out_type == 'opencv':
            self.order_trans = True
            self.div_factor = 1.0 / 255.0
        else:
            raise ValueError("Unknown key for {} {}".format(in_type, out_type))

    def __call__(self, img_tensor):

        if self.order_trans:
            img_tensor = img_tensor[:, [2, 1, 0], :, :].contiguous()
            img_tensor.div_(self.div_factor)

        for i in range(3):
            img_tensor[:, i, :, :].sub_(self.mean[i]).div_(self.std[i])

        return img_tensor


def crop_with_boxes(img_tensor, x_crop_boxes, out_height, out_width, crop_inds=None, avg_channels=True,
                    has_normed_coords=False):
    """Crop the image tensor by given boxes. The output will be resized to target size

    Params:
        img_tensor: torch.Tensor, in shape of [N, C, H, W]. If N > 1, the crop_inds must be specified.
        crop_boxes: list/numpy.ndarray/torch.Tensor in shape of [K x 4].
        out_height: int.
        out_width: int.
        crop_inds: list/numpy.ndarray/torch.Tensor in shape of [K]
    Returns:
        crop_img_tensor: torch.Tensor, in shape of [K, C, H, W]
    """

    img_device = img_tensor.device

    if isinstance(x_crop_boxes, list):
        crop_boxes = torch.tensor(x_crop_boxes, dtype=torch.float32).to(img_device)
    elif isinstance(x_crop_boxes, np.ndarray):
        crop_boxes = torch.tensor(x_crop_boxes, dtype=torch.float32).to(img_device)
    elif isinstance(x_crop_boxes, torch.Tensor):
        # change type and device if necessary
        crop_boxes = x_crop_boxes.clone().to(device=img_device, dtype=torch.float32)
    else:
        raise ValueError('Unknown type for crop_boxes {}'.format(type(x_crop_boxes)))

    if len(crop_boxes.size()) == 1:
        crop_boxes = crop_boxes.view(1, 4)

    num_imgs, chanenls, img_height, img_width = img_tensor.size()
    num_crops = crop_boxes.size(0)

    if crop_inds is not None:
        if isinstance(crop_inds, list) or isinstance(crop_inds, np.ndarray):
            crop_inds = torch.tensor(crop_inds, dtype=torch.float32).to(img_device)
        elif isinstance(crop_inds, torch.Tensor):
            crop_inds = crop_inds.to(device=img_device, dtype=torch.float32)
        else:
            raise ValueError('Unknown type for crop_inds {}'.format(type(crop_inds)))
        crop_inds = crop_inds.view(-1)
        assert crop_inds.size(0) == crop_boxes.size(0)
    else:
        if num_imgs == 1:
            crop_inds = torch.zeros(num_crops, dtype=torch.float32, device=img_device)
        elif num_imgs == num_crops:
            crop_inds = torch.arange(num_crops, dtype=torch.float32, device=img_device)
        else:
            raise ValueError('crop_inds MUST NOT be None.')

    if avg_channels:
        img_channel_avg = img_tensor.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        img_tensor_minus_avg = img_tensor - img_channel_avg  # minus mean values
    else:
        img_tensor_minus_avg = img_tensor
    crop_img_tensor = CropAndResizeFunction(out_height, out_width, has_normed=has_normed_coords)(
        img_tensor_minus_avg, crop_boxes, crop_inds)

    if avg_channels:
        # add mean value
        crop_img_tensor += img_channel_avg[crop_inds.long()]

    return crop_img_tensor



# -----------------------------------
# Functions for benchmark and others
# -----------------------------------
def load_dataset(dataset):
    """
    support OTB and VOT now
    TODO: add other datasets
    """
    info = {}

    if 'OTB' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]
            info[v]['name'] = v

    elif 'VOT' in dataset and (not 'VOT2019RGBT' in dataset) and (not 'VOT2020' in dataset):
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'VOT2020' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = open(gt_path, 'r').readlines()
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'RGBT234' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['infrared_imgs'] = [join(base_path, path_name, 'infrared', im_f) for im_f in
                                        info[v]['infrared_imgs']]
            info[v]['visiable_imgs'] = [join(base_path, path_name, 'visible', im_f) for im_f in
                                        info[v]['visiable_imgs']]
            info[v]['infrared_gt'] = np.array(info[v]['infrared_gt'])  # 0-index
            info[v]['visiable_gt'] = np.array(info[v]['visiable_gt'])  # 0-index
            info[v]['name'] = v

    elif 'VOT2019RGBT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            in_image_path = join(video_path, 'ir', '*.jpg')
            rgb_image_path = join(video_path, 'color', '*.jpg')
            in_image_files = sorted(glob.glob(in_image_path))
            rgb_image_files = sorted(glob.glob(rgb_image_path))

            assert len(in_image_files) > 0, 'please check RGBT-VOT dataloader'
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'infrared_imgs': in_image_files, 'visiable_imgs': rgb_image_files, 'gt': gt, 'name': video}
    elif 'VISDRONEVAL' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'annotations')
        attr_path = join(base_path, 'attributes')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'VISDRONETEST' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'initialization')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',').reshape(1, 4)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'GOT10KVAL' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.remove('list.txt')
        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'GOT10K' in dataset:  # GOT10K TEST
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.remove('list.txt')
        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': [gt], 'name': video}

    elif 'LASOT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        testingvideos = list(jsons.keys())

        father_videos = sorted(os.listdir(base_path))
        for f_video in father_videos:
            f_video_path = join(base_path, f_video)
            son_videos = sorted(os.listdir(f_video_path))
            for s_video in son_videos:
                if s_video not in testingvideos:  # 280 testing videos
                    continue

                s_video_path = join(f_video_path, s_video)
                # ground truth
                gt_path = join(s_video_path, 'groundtruth.txt')
                gt = np.loadtxt(gt_path, delimiter=',')
                gt = gt - [1, 1, 0, 0]
                # get img file
                img_path = join(s_video_path, 'img', '*jpg')
                image_files = sorted(glob.glob(img_path))

                info[s_video] = {'image_files': image_files, 'gt': gt, 'name': s_video}
    elif 'DAVIS' in dataset and 'TEST' not in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', 'DAVIS')
        list_path = join(realpath(dirname(__file__)), '../../dataset', 'DAVIS', 'ImageSets', dataset[-4:],
                         'val.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            info[video] = {}
            info[video]['anno_files'] = sorted(glob.glob(join(base_path, 'Annotations/480p', video, '*.png')))
            info[video]['image_files'] = sorted(glob.glob(join(base_path, 'JPEGImages/480p', video, '*.jpg')))
            info[video]['name'] = video
    elif 'YTBVOS' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', 'YTBVOS', 'valid')
        json_path = join(realpath(dirname(__file__)), '../../dataset', 'YTBVOS', 'valid', 'meta.json')
        meta = json.load(open(json_path, 'r'))
        meta = meta['videos']
        info = dict()
        for v in meta.keys():
            objects = meta[v]['objects']
            frames = []
            anno_frames = []
            info[v] = dict()
            for obj in objects:
                frames += objects[obj]['frames']
                anno_frames += [objects[obj]['frames'][0]]
            frames = sorted(np.unique(frames))
            info[v]['anno_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in frames]
            info[v]['anno_init_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in anno_frames]
            info[v]['image_files'] = [join(base_path, 'JPEGImages', v, im_f + '.jpg') for im_f in frames]
            info[v]['name'] = v

            info[v]['start_frame'] = dict()
            info[v]['end_frame'] = dict()
            for obj in objects:
                start_file = objects[obj]['frames'][0]
                end_file = objects[obj]['frames'][-1]
                info[v]['start_frame'][obj] = frames.index(start_file)
                info[v]['end_frame'][obj] = frames.index(end_file)

    else:
        raise ValueError("Dataset not support now, edit for other dataset youself...")

    return info

def load_video_info_im_gt(dataset, video_name):
    if 'LASOT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', dataset+'.json')
        jsons = json.load(open(json_path, 'r'))
        testingvideos = list(jsons.keys())


        father_video = video_name.split('-')[0]

        f_video_path = join(base_path, father_video)
        s_video_path = join(f_video_path, video_name)
        
        # ground truth
        gt_path = join(s_video_path, 'groundtruth.txt')
        gt = np.loadtxt(gt_path, delimiter=',')
        gt = gt - [1, 1, 0, 0]
        # get img file
        img_path = join(s_video_path, 'img', '*jpg')
        image_files = sorted(glob.glob(img_path))
                
        imgs = []
        for path in image_files:
            imgs.append(cv2.imread(path))

    else:
        raise ValueError('not supported now')

    return imgs, gt

def check_keys(model, pretrained_state_dict, print_unuse=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # remove num_batches_tracked
    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)

    print('missing keys:{}'.format(missing_keys))
    if print_unuse:
        print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters share common prefix 'module.'
    '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path, print_unuse=True):
    print('load pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')  # remove online train
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')  # remove multi-gpu label
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')   # remove online train

    check_keys(model, pretrained_dict, print_unuse=print_unuse)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def trans_model(model_path, save_path):
    pretrained = torch.load(model_path, map_location=lambda storage, loc: storage)
    save_ckpt = {}

    # # for self train imagenet
    # pretrained = remove_prefix(pretrained['state_dict'], 'module.')

    save_ckpt = {}
    for key in pretrained.keys():
        if key.startswith('layer'):
            key_in_new_res = 'features.features.' + key
            save_ckpt[key_in_new_res] = pretrained[key]

        else:
            save_ckpt[key] = pretrained[key]

    torch.save(save_ckpt, save_path)


Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')

def corner2center(corner):
    """
    [x1, y1, x2, y2] --> [cx, cy, w, h]
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h

def center2corner(center):
    """
    [cx, cy, w, h] --> [x1, y1, x2, y2]
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2

def IoU(rect1, rect2):
    # overlap

    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)

    target_a = (tx2-tx1) * (ty2 - ty1)

    inter = ww * hh
    overlap = inter / (area + target_a - inter)

    return overlap

def aug_apply(bbox, param, shape, inv=False, rd=False):
    """
    apply augmentation
    :param bbox: original bbox in image
    :param param: augmentation param, shift/scale
    :param shape: image shape, h, w, (c)
    :param inv: inverse
    :param rd: round bbox
    :return: bbox(, param)
        bbox: augmented bbox
        param: real augmentation param
    """
    if not inv:
        center = corner2center(bbox)
        original_center = center

        real_param = {}
        if 'scale' in param:
            scale_x, scale_y = param['scale']
            imh, imw = shape[:2]
            h, w = center.h, center.w

            scale_x = min(scale_x, float(imw) / w)
            scale_y = min(scale_y, float(imh) / h)
            center = Center(center.x, center.y, center.w * scale_x, center.h * scale_y)

        bbox = center2corner(center)

        if 'shift' in param:
            tx, ty = param['shift']
            x1, y1, x2, y2 = bbox
            imh, imw = shape[:2]

            tx = max(-x1, min(imw - 1 - x2, tx))
            ty = max(-y1, min(imh - 1 - y2, ty))

            bbox = Corner(x1 + tx, y1 + ty, x2 + tx, y2 + ty)

        if rd:
            bbox = Corner(*map(round, bbox))

        current_center = corner2center(bbox)

        real_param['scale'] = current_center.w / original_center.w, current_center.h / original_center.h
        real_param['shift'] = current_center.x - original_center.x, current_center.y - original_center.y

        return bbox, real_param
    else:
        if 'scale' in param:
            scale_x, scale_y = param['scale']
        else:
            scale_x, scale_y = 1., 1.

        if 'shift' in param:
            tx, ty = param['shift']
        else:
            tx, ty = 0, 0

        center = corner2center(bbox)

        center = Center(center.x - tx, center.y - ty, center.w / scale_x, center.h / scale_y)

        return center2corner(center)


# others
def cxy_wh_2_rect(pos, sz):
    return [float(max(float(0), pos[0]-sz[0]/2)), float(max(float(0), pos[1]-sz[1]/2)), float(sz[0]), float(sz[1])]  # 0-index


def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2

    return cx, cy, w, h

# poly_iou and _to_polygon comes from Linghua Huang
def poly_iou(polys1, polys2, bound=None):
    r"""Intersection over union of polygons.

    Args:
        polys1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
        polys2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
    """
    assert polys1.ndim in [1, 2]
    if polys1.ndim == 1:
        polys1 = np.array([polys1])
        polys2 = np.array([polys2])
    assert len(polys1) == len(polys2)

    polys1 = _to_polygon(polys1)
    polys2 = _to_polygon(polys2)
    if bound is not None:
        bound = box(0, 0, bound[0], bound[1])
        polys1 = [p.intersection(bound) for p in polys1]
        polys2 = [p.intersection(bound) for p in polys2]

    eps = np.finfo(float).eps
    ious = []
    for poly1, poly2 in zip(polys1, polys2):
        area_inter = poly1.intersection(poly2).area
        area_union = poly1.union(poly2).area
        ious.append(area_inter / (area_union + eps))
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _to_polygon(polys):
    r"""Convert 4 or 8 dimensional array to Polygons

    Args:
        polys (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
    """

    def to_polygon(x):
        assert len(x) in [4, 8]
        if len(x) == 4:
            return box(x[0], x[1], x[0] + x[2], x[1] + x[3])
        elif len(x) == 8:
            return Polygon([(x[2 * i], x[2 * i + 1]) for i in range(4)])

    if polys.ndim == 1:
        return to_polygon(polys)
    else:
        return [to_polygon(t) for t in polys]


def restore_from(model, optimizer, ckpt_path):
    print('restore from {}'.format(ckpt_path))
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path, map_location = lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']
    arch = ckpt['arch']
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch,  arch


def print_speed(i, i_time, n, logger):
    """print_speed(index, index_time, total_iteration)"""
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))
    logger.info('\nPROGRESS: {:.2f}%\n'.format(100 * i / n))  # for philly. let's reduce it in case others kill our job 100-25


def create_logger(cfg, modelFlag='OCEAN', phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    cfg = cfg[modelFlag]
    model = cfg.TRAIN.MODEL

    final_output_dir = root_output_dir / model

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = root_output_dir / model / (model + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    """
    save checkpoint
    """
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def save_model(model, epoch, optimizer, model_name, cfg, isbest=False):
    """
    save model
    """
    if not exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)

    if epoch > 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_name,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }, isbest, cfg.CHECKPOINT_DIR, 'checkpoint_e%d.pth' % (epoch + 1))
    else:
        print('epoch not save(<5)')


def extract_eaos(lines):
    """
    extract info of VOT eao
    """
    epochs = []
    eaos = []
    for line in lines:
        print(line)
        # if not line.startswith('[*]'):   # matlab version
        if not line.startswith('| Ocean'):
            continue
        temp = line.split('|')
        epochs.append(int(temp[1].split('_e')[-1]))
        eaos.append(float(temp[-2]))
    # fine bese epoch
    idx = eaos.index(max(eaos))
    epoch = epochs[idx]
    return epoch


def extract_logs(logfile, prefix):
    """
    extract logs for tuning, return best epoch number
    prefix: VOT, OTB, VOTLT, VOTRGBD, VOTRGBT
    """
    lines = open(logfile, 'r').readlines()
    if prefix == 'VOT':
        epoch = extract_eaos(lines)
    else:
        raise ValueError('not supported now')

    return 'checkpoint_e{}.pth'.format(epoch)


# ----------------------------
# build lr (from SiamRPN++)
# ---------------------------
class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr
                for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__,
                                             self.lr_spaces)


class LogScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.lr_spaces = np.logspace(math.log10(start_lr),
                                     math.log10(end_lr),
                                     epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 step=10, mult=0.1, epochs=50, last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        self.lr_spaces = self.start_lr * (mult**(np.arange(epochs) // step))
        self.mult = mult
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class MultiStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 steps=[10, 20, 30, 40], mult=0.5, epochs=50,
                 last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (len(steps)))
            else:
                mult = math.pow(end_lr/start_lr, 1. / len(steps))
        self.start_lr = start_lr
        self.lr_spaces = self._build_lr(start_lr, steps, mult, epochs)
        self.mult = mult
        self.steps = steps

        super(MultiStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, steps, mult, epochs):
        lr = [0] * epochs
        lr[0] = start_lr
        for i in range(1, epochs):
            lr[i] = lr[i-1]
            if i in steps:
                lr[i] *= mult
        return np.array(lr, dtype=np.float32)


class LinearStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = np.linspace(start_lr, end_lr, epochs)
        super(LinearStepScheduler, self).__init__(optimizer, last_epoch)


class CosStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = self._build_lr(start_lr, end_lr, epochs)

        super(CosStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, end_lr, epochs):
        index = np.arange(epochs).astype(np.float32)
        lr = end_lr + (start_lr - end_lr) * \
            (1. + np.cos(index * np.pi / epochs)) * 0.5
        return lr.astype(np.float32)


class WarmUPScheduler(LRScheduler):
    def __init__(self, optimizer, warmup, normal, epochs=50, last_epoch=-1):
        warmup = warmup.lr_spaces  # [::-1]
        normal = normal.lr_spaces
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)


LRs = {
    'log': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler}


def _build_lr_scheduler(optimizer, config, epochs=50, last_epoch=-1):
    return LRs[config.TYPE](optimizer, last_epoch=last_epoch,
                            epochs=epochs, **config.KWARGS)


def _build_warm_up_scheduler(optimizer, cfg, epochs=50, last_epoch=-1, modelFLAG='OCEAN'):
    #cfg = cfg[modelFLAG]
    warmup_epoch = cfg.TRAIN.WARMUP.EPOCH
    sc1 = _build_lr_scheduler(optimizer, cfg.TRAIN.WARMUP,
                              warmup_epoch, last_epoch)
    sc2 = _build_lr_scheduler(optimizer, cfg.TRAIN.LR,
                              epochs - warmup_epoch, last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_lr_scheduler(optimizer, cfg, epochs=50, last_epoch=-1, modelFLAG='OCEAN'):
    cfg = cfg[modelFLAG]
    if cfg.TRAIN.WARMUP.IFNOT:
        return _build_warm_up_scheduler(optimizer, cfg, epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer, cfg.TRAIN.LR, epochs, last_epoch)



# ----------------------------------
# Some functions for online
# ----------------------------------

## original utils/params.py
class TrackerParams:
    """Class for tracker parameters."""
    def free_memory(self):
        for a in dir(self):
            if not a.startswith('__') and hasattr(getattr(self, a), 'free_memory'):
                getattr(self, a).free_memory()


class FeatureParams:
    """Class for feature specific parameters"""
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError

        for name, val in kwargs.items():
            if isinstance(val, list):
                setattr(self, name, TensorList(val))
            else:
                setattr(self, name, val)


def Choice(*args):
    """Can be used to sample random parameter values."""
    return random.choice(args)




