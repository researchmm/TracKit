from os.path import join
from os import listdir
import json
import cv2
import numpy as np
from pprint import pprint

print('loading json (raw RGBT234 info), please wait 20 seconds~')
RGBT234 = json.load(open('RGBT234.json', 'r'))
RGBT234_base_path = '/data/zpzhang/datasets/dataset/RGBT234'

def check_size(frame_sz, bbox):
    min_ratio = 0.1
    max_ratio = 0.75
    # only accept objects >10% and <75% of the total frame
    area_ratio = np.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/float(np.prod(frame_sz)))
    ok = (area_ratio > min_ratio) and (area_ratio < max_ratio)
    return ok


def check_borders(frame_sz, bbox):
    dist_from_border = 0.05 * (bbox[2] - bbox[0] + bbox[3] - bbox[1])/2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and \
         ((frame_sz[0] - bbox[2]) > dist_from_border) and \
         ((frame_sz[1] - bbox[3]) > dist_from_border)
    return ok


snippets = dict()

n_videos = 0


for v_name in list(RGBT234.keys()):
    video = RGBT234[v_name]
    n_videos += 1
    in_frames = video['infrared_imgs']
    rgb_frames = video['visible_imgs']
    snippet = dict()
    snippets[video['name']] = dict()

    # read a image to get im size
    im_temp_path = join(RGBT234_base_path, video['name'], 'visible', rgb_frames[0])
    im_temp = cv2.imread(im_temp_path)
    frame_sz = [im_temp.shape[1], im_temp.shape[0]]

    in_gts = video['infrared_gt']
    rgb_gts = video['visible_gt']

    for f, in_frame in enumerate(in_frames):
        in_bbox = in_gts[f]  # (x,y,w,h)
        rgb_bbox = rgb_gts[f]  # (x,y,w,h)

        bboxs = [[in_bbox[0], in_bbox[1], in_bbox[0]+in_bbox[2], in_bbox[1]+in_bbox[3]],
                 [rgb_bbox[0], rgb_bbox[1], rgb_bbox[0]+rgb_bbox[2], rgb_bbox[1]+rgb_bbox[3]]]  #(xmin, ymin, xmax, ymax)

        imgs = [in_frames[f], rgb_frames[f]] # image name may be different in visible and rgb imgs

        snippet['{:06d}'.format(f)] = [imgs, bboxs]

    snippets[video['name']]['{:02d}'.format(0)] = snippet.copy()

json.dump(snippets, open('/data/share/SMALLSIAM/RGBT234/all.json', 'w'), indent=4, sort_keys=True)
print('done!')
