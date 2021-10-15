# Ocean tutorial
## Testing

We assume the root path is $TracKit, e.g. `/home/zpzhang/TracKit`
### Set up environment

```
cd $TracKit/lib/tutorial
bash install.sh $conda_path TracKit
cd $TracKit
conda activate TracKit
python setup.py develop
```
`$conda_path` denotes your anaconda path, e.g. `/home/zpzhang/anaconda3`

- **[Optional]** Install TensorRT according to the [tutorial](../install_trt.md).

**Note:** we perform TensorRT evaluation on RTX2080 Ti and CUDA10.0. If you fail to install it, please use pytorch version.



### Prepare data and models
1. Download the pretrained [PyTorch model](https://drive.google.com/drive/folders/1XU5wmyC7MsI6C_9Lv-UH1mwDIh57FFf8?usp=sharing) and [TensorRT model](https://github.com/researchmm/TracKit/releases/tag/tensorrt) to `$TracKit/snapshot`.
2. Download [json](https://drive.google.com/drive/folders/1kYX_c8rw7HMW0e5V400vaLy9huiYvDHE?usp=sharing) files of testing data and put them in `$TracKit/dataset`.
3. Download testing data e.g. VOT2019 and put them in `$TracKit/dataset`. Please download each data from their official websites, and the directories should be named like `VOT2019`, `OTB2015`, `GOT10K`, `LASOT`.

### Testing
In root path `$TracKit`,

```
python tracking/test_ocean.py --arch Ocean --resume snapshot/OceanV.pth --dataset VOT2019
```
### Evaluation
```
python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset VOT2019 --tracker_result_dir result/VOT2019 --trackers Ocean
```
You may test other datasets with our code. Please corresponds the provided pre-trained model `--resume` and dataset `--dataset`. See [ocean_model.txt](https://drive.google.com/file/d/1T2QjyxN4movpFtpzCH8xHHX5_Dz7G5Y6/view?usp=sharing) for their correspondences.


### TensorRT toy
Testing video: `twinnings` in OTB2015 (472 frames)
Testing GPU: `RTX2080Ti`

- TensorRT (**149fps**)
```
python tracking/test_ocean.py --arch OceanTRT --resume snapshot/OceanV.pth --dataset OTB2015 --video twinnings
```

- Pytorch (**68fps**)
```
python tracking/test_ocean.py --arch Ocean --resume snapshot/OceanV.pth --dataset OTB2015 --video twinnings
```

**Note:** 
- TensorRT version of Ocean only supports 255 input.
- Current TensorRT does not well support some operations. We would continuously renew it following official TensorRT updating. If you want to test on the benchmark, please us the Pytorch version. 
- If you want to use our code in a realistic product, our TensorRT code may help you.



:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:
## Training
#### prepare data
- Please download training data from [GoogleDrive](https://drive.google.com/drive/folders/1ehjVhg6ewdWSWt709zd1TkjWF7UJlQlq?usp=sharing) or [BaiduDrive(urxq)](https://pan.baidu.com/s/1jGPEJieir5OWqCmibV3yrQ), and then put them in `$TracKit/data`
- You could also refer to scripts in `$TracKit/lib/dataset/crop` to process your custom data. 
- For splited files in BaiduDrive, please use `cat got10k.tar.*  | tar -zxv` to merge and unzip.


#### prepare pretrained model
Please download the pretrained model on ImageNet [here](https://drive.google.com/drive/folders/1ctoxaPiS9qinhmN_bl5z3VNhYnrhl99t?usp=sharing), and then put it in `$TracKit/pretrain`.

#### modify settings
Please modify the training settings in `$TracKit/experiments/train/Ocean.yaml`. The default number of GPU and batch size in paper are 8 and 32 respectively. 

#### run
In root path $TracKit,
```
python tracking/onekey.py
```
This script integrates **train**, **epoch test** and **tune**. It is suggested to run them one by one when you are not familiar with our whole framework (modify the key `ISTRUE` in `$TracKit/experiments/train/Ocean.yaml`). When you know this framework well, simply run this one-key script. VOT2018 is much more sensitive than other datasets, thus I would suggest you tune 4000-5000 groups for it. For other datasets like VOT2019/OTB, 1500-2000 may be enough. For truely large dataset like LASOT, I would suggest you tune with grid search (only selecting epoch and tuning `window_influence` is enough for LASOT in my experience.)
