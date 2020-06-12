# --------------------------------------------------------
# process msra10k
# --------------------------------------------------------
import os
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import argparse
import pdb
from PIL import Image

parser = argparse.ArgumentParser(description='COCO Parallel Preprocessing for SiamMask')
parser.add_argument('--exemplar_size', type=int, default=127, help='size of exemplar')
parser.add_argument('--context_amount', type=float, default=0.5, help='context amount')
parser.add_argument('--search_size', type=int, default=511, help='size of cropped search region')
parser.add_argument('--enable_mask', action='store_true', help='whether crop mask')
parser.add_argument('--num_threads', type=int, default=24, help='number of threads')
args = parser.parse_args()


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFCx(image, bbox, exemplar_size=127, context_amount=0.5, search_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return x


def crop_img(video_name, set_crop_base_path, set_img_base_path,
             exemplar_size=127, context_amount=0.5, search_size=511, enable_mask=True):
    video_name = video_name[:-1]
    imgs = sorted(os.listdir(join(set_img_base_path, 'JPEGImages/480p', video_name)))

    for im_id, im_name in enumerate(imgs):
        im_path = join(set_img_base_path, 'JPEGImages/480p', video_name, im_name)
        mask_path = join(set_img_base_path, 'Annotations/480p', video_name, im_name.replace('.jpg', '.png'))
        im = cv2.imread(im_path)
        avg_chans = np.mean(im, axis=(0, 1))
        mask = np.array(Image.open(mask_path)).astype(np.uint8)

        objects = np.unique(mask)
        frame_crop_base_path = join(set_crop_base_path, video_name)  # video path
        if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

        for track_id in range(1, len(objects)):
            
            color = objects[track_id]
            mask_temp = (mask == color).astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(mask_temp)
            bbox = [x, y, x + w - 1, y + h - 1] # [x1,y1,x2,y2]

            x = crop_like_SiamFCx(im, bbox, exemplar_size=exemplar_size, context_amount=context_amount,
                                  search_size=search_size, padding=avg_chans)
            cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(im_id, track_id-1)), x)

            x = crop_like_SiamFCx(mask_temp, bbox, exemplar_size=exemplar_size, context_amount=context_amount, search_size=search_size)
            x  = (x > 0.5 * 255).astype(np.uint8) * 255
            cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.m.png'.format(im_id, track_id-1)), x)


def main(exemplar_size=127, context_amount=0.5, search_size=511, enable_mask=True, num_threads=24):
    global coco  # will used for generate mask
    data_dir = '/home/zpzhang/data/testing/DAVIS-trainval'
    crop_path = '/home/zpzhang/data/training/DAVIS/crop{:d}'.format(search_size)
    if not isdir(crop_path): makedirs(crop_path)

    train_txt = join(data_dir, 'ImageSets/2017', 'train.txt')
    videos = open(train_txt, 'r').readlines()
    n_videos = len(videos)
    
    # debug
    # for video in videos:
    #     if not video == 'cat-girl\n':
    #         continue
    #     crop_img(video, crop_path, data_dir, exemplar_size, context_amount, search_size, enable_mask)

    #
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(crop_img, video,
                                crop_path, data_dir,
                                exemplar_size, context_amount, search_size,
                                enable_mask) for video in videos]
        for i, f in enumerate(futures.as_completed(fs)):
            printProgress(i, n_videos, prefix='DAVIS', suffix='Done ', barLength=40)
    print('done')


if __name__ == '__main__':
    since = time.time()
    main(args.exemplar_size, args.context_amount, args.search_size, args.enable_mask, args.num_threads)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
