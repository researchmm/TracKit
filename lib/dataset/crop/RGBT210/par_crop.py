from os.path import join, isdir, exists
from os import listdir, mkdir, makedirs
import cv2
import numpy as np
import glob
from concurrent import futures
import sys
import time


RGBT234_base_path = '/data/share/RGBT210'

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
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]   # width, height
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_img(im, bbox, instanc_size):
    avg_chans = np.mean(im, axis=(0, 1))
    z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
    return z, x


eps = 1e-5
def crop_video(video, crop_path, instanc_size):
    video_crop_base_path = join(crop_path, video)
    if not exists(video_crop_base_path): makedirs(video_crop_base_path)

    video_base_path = join(RGBT234_base_path, video)

    # infrared gt
    in_gts_path = join(video_base_path, 'init.txt')
    try:
        in_gts = np.loadtxt(open(in_gts_path, "rb"), delimiter=',')
    except:
        in_gts = np.loadtxt(open(in_gts_path, "rb"), delimiter=' ')

    # rgb gt
    rgb_gts_path = join(video_base_path, 'init.txt')
    try:
        rgb_gts = np.loadtxt(open(rgb_gts_path, "rb"), delimiter=',')
    except:
        rgb_gts = np.loadtxt(open(rgb_gts_path, "rb"), delimiter=' ')

    in_jpgs = sorted(glob.glob(join(video_base_path, 'infrared', '*.jpg')))
    rgb_jpgs = sorted(glob.glob(join(video_base_path, 'visible', '*.jpg')))


    for idx, img_path in enumerate(in_jpgs):
        in_im = cv2.imread(img_path)
        rgb_im = cv2.imread(rgb_jpgs[idx])
        in_gt = in_gts[idx]
        rgb_gt = rgb_gts[idx]
        in_bbox = [int(g) for g in in_gt]  # (x,y,w,h)

        if abs(in_bbox[2]) < eps or abs(in_bbox[3]) < eps:
            continue

        in_bbox = [in_bbox[0], in_bbox[1], in_bbox[0]+in_bbox[2], in_bbox[1]+in_bbox[3]]   # (xmin, ymin, xmax, ymax)
        rgb_bbox = [int(g) for g in rgb_gt]  # (x,y,w,h)
        rgb_bbox = [rgb_bbox[0], rgb_bbox[1], rgb_bbox[0] + rgb_bbox[2], rgb_bbox[1] + rgb_bbox[3]]  # (xmin, ymin, xmax, ymax)

        in_z, in_x = crop_img(in_im, in_bbox, instanc_size)
        rgb_z, rgb_x = crop_img(rgb_im, rgb_bbox, instanc_size)

        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.in.z.jpg'.format(int(idx), 0)), in_z)
        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.in.x.jpg'.format(int(idx), 0)), in_x)

        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.rgb.z.jpg'.format(int(idx), 0)), rgb_z)
        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.rgb.x.jpg'.format(int(idx), 0)), rgb_x)


def main(instanc_size=271, num_threads=24):
    crop_path = '/data/share/SMALLSIAM/RGBT210/crop{:d}'.format(instanc_size)
    if not exists(crop_path): makedirs(crop_path)

    videos = sorted(listdir(RGBT234_base_path))
    n_videos = len(videos)

    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(crop_video, video, crop_path, instanc_size) for video in videos]
        for i, f in enumerate(futures.as_completed(fs)):
            # Write progress to error so that it can be seen
            printProgress(i, n_videos, prefix='RGBT210', suffix='Done ', barLength=40)


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
