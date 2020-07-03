from os.path import join
from os import listdir
import json
import numpy as np

print('loading json (raw visdrone info), please wait 20 seconds~')
visdrone = json.load(open('visdrone.json', 'r'))


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
for subset in visdrone:
    for video in subset:
        n_videos += 1
        frames = video['frame']
        snippet = dict()
        bp = video['base_path']
        bp = bp.split('/')
        bp = join(bp[0], bp[-1])

        snippets[bp] = dict()
        for f, frame in enumerate(frames):
            frame_sz = frame['frame_sz']
            bbox = frame['bbox']  # (x,y,w,h)

            snippet['{:06d}'.format(f)] = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]   #(xmin, ymin, xmax, ymax)

        snippets[bp]['{:02d}'.format(0)] = snippet.copy()
        
# train = {k:v for (k,v) in snippets.items() if 'train' in k}
# val = {k:v for (k,v) in snippets.items() if 'val' in k}

train = {k:v for (k,v) in snippets.items()}

# json.dump(train, open('/data2/visdrone/train.json', 'w'), indent=4, sort_keys=True)
json.dump(train, open('/data/home/v-zhipeng/dataset/training/VISDRONE/train.json', 'w'), indent=4, sort_keys=True)
print('done!')
