# Adaptive Anchor-Free Tracking

## Introduction
Anchor-based Siamese trackers have witnessed tremendous success in visual tracking, yet their robustness still leaves a gap with recent state-of-the-art approaches. We find the underlying reason is that the regression network in anchor-based methods is only trained on the positive anchor boxes. This mechanism makes it difficult to refine the anchors whose overlap with the target objects are small. 
  
Our proposals mitigate the problem by,
1) We propose a novel anchor-free algorithm for realtime visual tracking. It is capable of rectifying the imprecise bounding-box predictions whose overlap with the target objects are small.

2)  We introduce an adaptive feature alignment mechanism to set up a correspondence between the objectaware features and the predicted bounding boxes. This leads to a more robust classification of foreground objects and background clutters.


<div align="center">
  <img src="demo/AdaFree.gif" width="800px" />
  <!-- <p>Example SiamFC, SiamRPN and SiamMask outputs.</p> -->
</div>

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:
## Results

| Models  | VOT18 | VOT19 | OTB15 | GOT10K | LASOT| 
| :------ | :------: | :------: | :------: | :------: | :------: |
| Offline  | 0.467 | 0.327 | 0.671 | 0.592 | 0.526 |  
| Online |  |  |  |  | |  
| Raw Results| | |  |  | |  



:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:
## Quick Start
### Installation

```
bash install.sh adafree
```

### Preparation
The test datasets should be arranged in `dataset` directory, and the pre-trained models should be arranged in `snapshot` directory.

```
${Tracking_ROOT}
|—— experimnets
|—— lib
|—— snapshot
  |—— xxx.model
|—— dataset
  |—— OTB2015.json
  |—— VOT2019.json 
  |—— VOT2019 (or OTB2015...)
     |—— videos...
|—— ...

```
Download [pre-trained models](https://drive.google.com/drive/folders/1nkSTnyLQidpW67AdD8T7BVsbgrY2_iYt?usp=sharing) and [json](https://drive.google.com/drive/folders/10hDmCLLo0c5Hs12kqB--Ctj95UTiFVrH?usp=sharing) files here.


### Test
```
python adafree_tracking/test_adafree.py --arch AdaFree --dataset VOT2019 --epoch_test False
```

### Evaluation
```
python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset VOT2019 --tracker_result_dir result/VOT2019 --trackers AdaFree
python ./lib/core/eval_otb.py OTB2015 ./result Ada* 0 1
```



 
<!-- ## Parameter Tuning Toolkit :tada::tada:
Lift is short, let's release our hands and brain. With our toolkit, you don't need to analysz how each hyper-parameter influence the final result. **Just a quick click!!** 
```
Coming Soon...
```
Now you can go to bed for a good sleep. The favorable results will come with dawn. -->
:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:
## Citation
If any part of our paper and code is helpful to your work, please generously cite with:

<!-- ```
@inproceedings{Zhang_2019_CVPR,
    author={Zhang, Zhipeng and Peng, Houwen},
    title={Deeper and Wider Siamese Networks for Real-Time Visual Tracking},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
``` -->

## License
Licensed under an MIT license.




