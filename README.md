
# TracKit

### This is a toolkit for video object tracking and segmentation.

<div align="left">
  <img src="demo/ocean1.gif" width="800px" />
  <!-- <p>Example SiamFC, SiamRPN and SiamMask outputs.</p> -->
</div>

## News
:boom: Hiring research interns for visual tracking, segmentation and neural architecture search projects: houwen.peng@microsoft.com

:boom: We achieves the runner-ups for both [VOT2020ST (short-term) and RT(real-time)](http://data.votchallenge.net/vot2020/presentations/vot2020-st.pdf). The variants of Ocean take **2nd/3rd/5th** places of VOT2020RT. The [SiamDW-T](https://github.com/researchmm/VOT2019) submitted to VOT2019 achieves **1st** of [VOT2020RGBT](http://data.votchallenge.net/vot2020/presentations/vot2020-rgbt.pdf) (submitted by VOT committee).  

:boom: Our paper [Ocean](https://arxiv.org/pdf/2006.10721v2.pdf) has been accepted by [ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3872_ECCV_2020_paper.php). 

:boom:  The initial version is released, including [Ocean(ECCV2020)](https://arxiv.org/pdf/2006.10721v2.pdf) and [SiamDW(CVPR2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf).

:boom: We provide a TensorRT implementation, running at 1.5~2.5 times faster than pytorch version (e.g. 149fps/68fps for video `twinnings`, see [details](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/)).

**Note:** We focus on providing an easy-to-follow code based on **Pytorch** and **TensorRT** for research on video object tracking and segmentation task. The code will be continuously optimized. You may pull requests to help us build this repo. 


<!-- 
## :+1: Recommendations 
:fire: Welcome to subscribe our [YouTube Channel](https://www.youtube.com/channel/UCrN0FSb26nsCP41ZJYo4Xpg). 

:fire: [Comparision:](https://github.com/JudasDie/Comparision) We summarize the performances of **97 trackers** (published in CVPR/ICCV/ECCV/AAAI/NIPS) on **15 tracking benchmarks** (OTB13/15, VOT16-20, LASOT, GOT10K, TrackingNet, UAV123, NFS, TC128, VOT2018LT, OxUvA).  The [repo.](https://github.com/JudasDie/Comparision) is designed to easily compare different trackers, especially when writing papers (performance table/figures). We will continuously update that repo., and we welcome your PR.
:fire: We provide some raw scripts used in our daily research. Some of them may be useful for your daily research. See [ResearchTools](https://github.com/JudasDie/ResearchTools). -->


## Trackers

### Ocean

**[[Paper]](https://arxiv.org/abs/2006.10721) [[Raw Results]](https://drive.google.com/drive/folders/1w_SifcV_Ddu2TSqaV-14XSgLlKZq2lPN?usp=sharing) [[Training and Testing]](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/Ocean/ocean.md) [[Demo]](https://www.youtube.com/watch?v=83-XCEsQ1Kg&feature=youtu.be)** <br/>

Official implementation of the Ocean tracker. Ocean proposes a general anchor-free based tracking framework. It includes a pixel-based anchor-free regression network to solve the weak rectification problem of RPN, and an object-aware classification network to learn robust target-related representation. Moreover, we introduce an effective multi-scale feature combination module to replace heavy result fusion mechanism in recent Siamese trackers. An additional **TensorRT** toy demo is provided in this repo.

<div align="left">
  <img src="https://github.com/penghouwen/TracKit/blob/master/demo/Ocean_overview.jpg" height="300" alt="Ocean"/><br/>
  <!-- <p>Example SiamFC, SiamRPN and SiamMask outputs.</p> -->
</div>

<!-- ### OceanPlus
Paper and Code will be released soon.

- **Advantages:** only several lines of code (core part) on Ocean, easy to implement.

- VOT2020 performances

| Models | Baseline | Realtime |
| --| --| --|
|Offline | 0.444 | 0.436 |
|Online  | 0.500 | 0.484 |

<div align="left">
  <img src="demo/oceanplus.gif" width="600px" />
  <!-- <p>Example SiamFC, SiamRPN and SiamMask outputs.</p> -->
<!-- </div>
<img src="https://github.com/penghouwen/TracKit/blob/master/demo/lines.jpg"  alt="Ocean"/><br/> --> 

### SiamDW
**[[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf) [[Raw Results]](https://github.com/researchmm/SiamDW) [[Training and Testing]](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/SiamDW/siamdw.md) [[Demo]]()** <br/>
SiamDW is one of the pioneering work using deep backbone networks for Siamese tracking framework. Based on sufficient analysis on network depth, output size, receptive field and padding mode, we propose guidelines to build backbone networks for Siamese tracker. Several deeper and wider networks are built following the guidelines with the proposed CIR module. 

<img src="https://github.com/penghouwen/TracKit/blob/master/demo/siamdw_overview.jpg" height="250" alt="SiamDW"/><br/>


<!-- </div>
### OceanPlus
**[[Paper]](https://arxiv.org/pdf/2008.02745v2.pdf) [[Raw Results]](https://drive.google.com/drive/folders/1doQiv82swum2rEXXo5C735WrLb_uAVbq?usp=sharing) [[Training and Testing Tutorial]](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/OceanPlus/oceanplus.md) [[Demo]](https://github.com/researchmm/TracKit/tree/master/demo/oceanplus.gif)** <br/>
Official implementation of the OceanPlus tracker. It proposes an attention retrieval network (ARN) to perform soft spatial constraints on backbone features. Concretely, we first build a look-up-table (LUT) with the ground-truth mask in the starting frame, and then retrieve the LUT to obtain a target-aware attention map for suppressing the negative influence of pixel-wise background clutter. Furthermore, we introduce a multi-resolution multi-stage segmentation network (MMS) to ulteriorly weaken responses of background clutter by reusing the predicted mask to filter backbone features.

</div>
<img src="https://github.com/researchmm/TracKit/blob/master/demo/oceanplu_overview.png" height="250" alt="OceanPlus"/><br/>
</div>
--> 

## How To Start

- Tutorial for **Ocean**

  Follow Ocean **[[Training and Testing]](https://github.com/researchmm/TracKit/blob/master/lib/tutorial/Ocean/ocean.md)** tutorial 

- Tutorial for **SiamDW**

  Follow SiamDW **[[Training and Testing]](https://github.com/researchmm/TracKit/blob/master/lib/tutorial/SiamDW/siamdw.md)** tutorial 


<!-- </div>
- Tutorial for **OceanPlus**

  Follow OceanPlus **[[Training and Testing]](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/OceanPlus/oceanplus.md)** tutorial
--> 

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
  |—— DAVIS
     |—— blackswan...
|—— ...

```


## ToDO
:anchor: Add testing/training code of other trackers.


## Citation
If any part of our paper or code helps your work, please generouslly cite our work:
```

@InProceedings{Ocean_2020_ECCV,
author = {Zhipeng Zhang, Houwen Peng, Jianlong Fu, Bing Li, Weiming Hu},
title = {Ocean: Object-aware Anchor-free Tracking},
booktitle = {European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
} 

@InProceedings{SiamDW_2019_CVPR,
author = {Zhang, Zhipeng and Peng, Houwen},
title = {Deeper and Wider Siamese Networks for Real-Time Visual Tracking},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
} 

@InProceedings{TVOS_2020_CVPR,
author = {Zhang, Yizhuo and Wu, Zhirong and Peng, Houwen and Lin, Stephen},
title = {A Transductive Approach for Video Object Segmentation},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

@article{OceanPlus_arxiv_2020,
  title={Towards Accurate Pixel-wise Object Tracking by Attention Retrieval},
  author={Zhipeng Zhang, Bing Li, Weiming Hu, Houwen Peng},
  journal={arXiv preprint arXiv:2001.10883},
  year={2020}
}

```

## References
```
[1] Bhat G, Danelljan M, et al. Learning discriminative model prediction for tracking. ICCV2019.
[2] Chen, Kai and Wang, et.al. MMDetection: Open MMLab Detection Toolbox and Benchmark.
[3] Li, B., Wu, W., Wang, Q., et.al. Siamrpn++: Evolution of siamese visual tracking with very deep networks. CVPR2019.
[4] Dai, J., Qi, H., Xiong, Y., et.al. Deformable convolutional networks. ICCV2017.
[5] Wang, Q., Zhang, L., et.al. Fast online object tracking and segmentation: A unifying approach. CVPR2019.
[6] Vu, T., Jang, H., et.al. Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution. NIPS2019.
[7] VOT python toolkit: https://github.com/StrangerZhang/pysot-toolkit 
```







