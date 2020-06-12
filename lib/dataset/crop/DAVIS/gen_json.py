# --------------------------------------------------------
# processing DAVIS train
# --------------------------------------------------------
from os.path import join
import json
import os
import cv2
import pdb
import numpy as np
import pdb
from PIL import Image

data_dir = '/home/zpzhang/data/testing/DAVIS-trainval'
saveDir = '/home/zpzhang/data/training/DAVIS'

dataset = dict()
train_txt = join(data_dir, 'ImageSets/2017', 'train.txt')
videos = open(train_txt, 'r').readlines()
n_videos = len(videos)

for iidx, video_name in enumerate(videos):
    video_name = video_name[:-1]

    print('video id: {:04d} / {:04d}'.format(iidx, n_videos))
    try:
        imgs = sorted(os.listdir(join(data_dir, 'JPEGImages/480p', video_name)))
    except:
        continue
    dataset[video_name] = dict()

    for idx, im_name in enumerate(imgs):
        mask_path = join(data_dir, 'Annotations/480p', video_name, im_name.replace('.jpg', '.png'))
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        objects = np.unique(mask)

        for track_id in range(1, len(objects)):
            color = objects[track_id]
            mask_temp = (mask == color).astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(mask_temp)
            bbox = [x, y, x + w - 1, y + h - 1] # [x1,y1,x2,y2]
            if w <= 0 or h <= 0:  # lead nan error in cls.
                continue

            if '{:02d}'.format(track_id - 1) not in dataset[video_name].keys():
                dataset[video_name]['{:02d}'.format(track_id - 1)] = dict()
            dataset[video_name]['{:02d}'.format(track_id-1)]['{:06d}'.format(int(im_name.split('.')[0]))] = bbox
print('save json (dataset), please wait 20 seconds~')
save_path = join(saveDir, 'davis.json')
json.dump(dataset, open(save_path, 'w'), indent=4, sort_keys=True)
print('done!')

