# Ocean tutorial
## Testing
### Set up environment
```
cd lib/tutorial
bash install.sh $conda_path TrackSeg
cd ../../
conda activate TrackSeg
python setup.py develop
```
- Install TensorRT according to the [tutorial](../install_trt.md).
**Note:** we perform TensorRT evaluation on RTX2080 Ti and CUDA10.0. If you fail to install it, please use pytorch version.



### Prepare data and models
1. Download the pretrained [PyTorch model](https://drive.google.com/drive/folders/1DfiuFP2xuclVLzPkPKYkMWJXHKAZLJmk?usp=sharing) and [TensorRT model](https://drive.google.com/file/d/1oY-8Qe5-QQIwEzh6V6MfyZGkTW8Esk-9/view?usp=sharing) to `snapshot`.
2. Download [json](https://drive.google.com/open?id=1S-RkzyMVRFWueWW91NmZldUJuDyhGdp1) files of testing data and put thme in `dataset`.
3. Download testing data e.g. VOT2019 and put them in `dataset`. 

### Testing
In root path `$TrackSeg`,
- PyTorch
```
python tracking/test_ocean.py --arch Ocean --resume snapshot/OceanV.pth --dataset VOT2019
```
### Evaluation
```
python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset VOT2019 --tracker_result_dir result/VOT2019 --trackers Ocean
```
You may test other datasets with our code. Please corresponds the provided pre-trained model `--resume` and dataset `--dataset`. See `ocean_model.txt` for their correspondences.


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
Please download training data from [here](https://drive.google.com/drive/folders/1ehjVhg6ewdWSWt709zd1TkjWF7UJlQlq?usp=sharing).

#### prepare pretrained model
Please download the pretrained model on ImageNet [here](https://drive.google.com/open?id=1Pwe5NRdOoGiTYlnrOZdL-3S494RkbPQe), and then put it in `pretrain`.

#### modify settings
Please modify the training settings in `experiments/train/Ocean.yaml`. The default number of GPU and batch size in paper are 8 and 32 respectively. 

#### run
```
python tracking/onekey.py
```
This script integrates **train**, **epoch test** and **tune**. It is suggested to run them one by one when you are not familiar with our whole framework (modify the key `ISTRUE` in `experiments/train/Ocean.yaml`). When you know this framework well, simply intergrate your idea and run this one-key script.
