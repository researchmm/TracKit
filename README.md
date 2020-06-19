
### Hiring research interns for visual tracking, segmentation and neural architecture search projects: houwen.peng@microsoft.com

# TracKit

### This is a toolkit for video object tracking and segmentation.

## News
:boom: Implementation of our tracking and segmentation work, based on **Pytorch** and **TensorRT**.

:boom:  The initial version is released, including [Ocean](https://arxiv.org/abs/2006.10721) and [SiamDW(CVPR2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf).

:boom: We provide a TensorRT implementation, running at 1.5~2.5 times faster than pytorch version (e.g. 149fps/68fps for video `twinnings`, see [details](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/)).

**Note:** We focus on providing an easy-to-follow code for research on video object tracking and segmentation task. The code will be continuously optimized. You may pull requests to help us build this repo. 




## ToDO
:anchor: Release paper and code for [OceanPlus]().



## Trackers
### OceanPlus
**[[Paper]]() [[Raw Results]](https://drive.google.com/drive/folders/1Wi_9fvPhfBtuDPteNo75iqiRs4-bfmgI?usp=sharing) [[Training and Testing]](https://github.com/JudasDie/TrackSeg/tree/master/lib/tutorial/OceanPlus) [[Demo]](https://www.youtube.com/watch?v=ueNqnVkl37c)** <br/>

- **Advantages:** only several lines of code (core part) on Ocean, easy to implement.

- VOT2020 performances

| Models | Baseline | Realtime |
| --| --| --|
|Offline | 0.444 | 0.436 |
|Online  | 0.500 | 0.484 |

<div align="left">
  <img src="demo/oceanplus.gif" width="600px" />
  <!-- <p>Example SiamFC, SiamRPN and SiamMask outputs.</p> -->
</div>
<img src="https://github.com/penghouwen/TracKit/blob/master/demo/lines.jpg"  alt="Ocean"/><br/>


### Ocean
**[[Paper]](https://arxiv.org/abs/2006.10721) [[Raw Results]](https://drive.google.com/file/d/1vDp4MIkWzLVOhZ-Yt2Zdq8Z_Z0rz6y0R/view?usp=sharing) [[Training and Testing]](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/Ocean) [[Demo]](https://www.youtube.com/watch?v=83-XCEsQ1Kg&feature=youtu.be)** <br/>
Official implementation of the Ocean tracker. Ocean proposes a general anchor-free based tracking framework. It includes a pixel-based anchor-free regression network to solve the weak rectification problem of RPN, and an object-aware classification network to learn robust target-related representation. Moreover, we introduce an effective multi-scale feature combination module to replace heavy result fusion mechanism in recent Siamese trackers. This work also serves as the baseline model of OceanPlus. An additional **TensorRT** toy demo is provided in this repo.
<img src="https://github.com/penghouwen/TracKit/blob/master/demo/Ocean_overview.jpg" height="300" alt="Ocean"/><br/>

### SiamDW
**[[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf) [[Raw Results]](https://github.com/researchmm/SiamDW) [[Training and Testing]](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/SiamDW) [[Demo]]()** <br/>
SiamDW is one of the pioneering work using deep backbone networks for Siamese tracking framework. Based on sufficient analysis on network depth, output size, receptive field and padding mode, we propose guidelines to build backbone networks for Siamese tracker. Several deeper and wider networks are built following the guidelines with the proposed CIR module. 

<img src="https://github.com/penghouwen/TracKit/blob/master/demo/siamdw_overview.jpg" height="250" alt="SiamDW"/><br/>


## Structure
- `experiments:` training and testing settings
- `demo:` figures for readme
- `dataset:` testing dataset
- `data:` training dataset
- `lib:` core scripts for all trackers
- `snapshot:` pre-trained models 
- `pretrain:` models trained on ImageNet (for training)
- `tutorials:` guidelines for training and testing
- `tracking:` training and testing interface

```
$TrackSeg
|—— experimnets
|—— lib
|—— snapshot
  |—— xxx.model/xxx.pth
|—— dataset
  |—— VOT2019.json 
  |—— VOT2019
     |—— ants1...
  |—— VOT2020
     |—— ants1...
|—— ...

```
Please follow **[[Training and Testing]](https://github.com/researchmm/TracKit/tree/master/lib/tutorial)** of each model to use our trackers. 

:dart: Further discussion anbout our paper and code: zhangzhipeng2017@ia.ac.cn

## Citation
If any part of our paper or code helps your work, please generouslly cite our work:
```
@InProceedings{SiamDW_2019_CVPR,
author = {Zhang, Zhipeng and Peng, Houwen},
title = {Deeper and Wider Siamese Networks for Real-Time Visual Tracking},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
} 

@article{zhangOcean,
  title={Ocean: Object-aware Anchor-free Tracking},
  author={Zhipeng Zhang, Houwen Peng, Jianlong Fu, Bing Li, Weiming Hu},
  journal={arXiv preprint arXiv:2006.10721
  year={2020}
}
```

## References
```
[1] Bhat G, Danelljan M, et al. Learning discriminative model prediction for tracking. ICCV2019.
[2] Chen, Kai and Wang, et.al. MMDetection: Open MMLab Detection Toolbox and Benchmark.
```
## Contributors
- **[Zhipeng Zhang](https://github.com/JudasDie)**
- **[Houwen Peng](https://houwenpeng.com/)**








