from os.path import join
from os import listdir
import json
import numpy as np

print('loading json (raw lasot info), please wait 20 seconds~')
lasot = json.load(open('lasot.json', 'r'))


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
for subset in lasot:
    for video in subset:
        n_videos += 1
        frames = video['frame']
        snippet = dict()

        snippets[video['base_path'].split('/')[-1]] = dict()
        for f, frame in enumerate(frames):
            frame_sz = frame['frame_sz']
            bbox = frame['bbox']  # (x,y,w,h)

            snippet['{:06d}'.format(f)] = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]   #(xmin, ymin, xmax, ymax)

        snippets[video['base_path'].split('/')[-1]]['{:02d}'.format(0)] = snippet.copy()

json.dump(snippets, open('/data/share/LASOT/train.json', 'w'), indent=4, sort_keys=True)
print('done!')
