from os.path import join, isdir, exists
from os import listdir, mkdir, makedirs
import cv2
import numpy as np
import glob
from concurrent import futures
import sys
import time


visdrone_base_path = '/data/home/v-zhipeng/dataset/testing/VISDRONE'
sub_sets = sorted({'VisDrone2019-SOT-train', 'VisDrone2019-SOT-val'})


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


def crop_video(sub_set, video, crop_path, instanc_size):
    video_crop_base_path = join(crop_path, sub_set, video)
    if not exists(video_crop_base_path): makedirs(video_crop_base_path)

    sub_set_base_path = join(visdrone_base_path, sub_set)
    video_base_path = join(sub_set_base_path, 'sequences', video)
    gts_path = join(sub_set_base_path, 'annotations', '{}.txt'.format(video))
    gts = np.loadtxt(open(gts_path, "rb"), delimiter=',')
    jpgs = sorted(glob.glob(join(video_base_path, '*.jpg')))

    if not jpgs:
        print('no jpg files, try png files')
        jpgs = sorted(glob.glob(join(video_base_path, '*.png')))
        if not jpgs:
            print('no jpg and png files, check data please')

    for idx, img_path in enumerate(jpgs):
        im = cv2.imread(img_path)
        avg_chans = np.mean(im, axis=(0, 1))
        gt = gts[idx]
        bbox = [int(g) for g in gt]  # (x,y,w,h)
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]   # (xmin, ymin, xmax, ymax)

        z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(idx), 0)), z)
        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(idx), 0)), x)


def main(instanc_size=511, num_threads=24):
    crop_path = '/data/home/v-zhipeng/dataset/training/VISDRONE/crop{:d}'.format(instanc_size)
    if not exists(crop_path): makedirs(crop_path)

    for sub_set in sub_sets:
        sub_set_base_path = join(visdrone_base_path, sub_set)
        videos = sorted(listdir(join(sub_set_base_path, 'sequences')))
        n_videos = len(videos)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, sub_set, video, crop_path, instanc_size) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
