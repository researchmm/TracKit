# OceanPlus tutorial
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


**Note:**  all the results for VOT2020 in the paper (including other methods) are performed with `vot-toolkit=0.2.0`. Please use the same env to reproduce our results.


### Prepare data and models

1. Following the official [guidelines](https://www.votchallenge.net/howto/tutorial_python.html) to set up VOT workspace.

2. Download from [GoogleDrive](https://drive.google.com/drive/folders/1_uagYRFpQmYoWAc0oeiAY49gHwQxztrN?usp=sharing) and put them in `$TracKit/snapshot`


### Testing

#### For VOT2020

- **Note: the results are 0.444/0.451 for single stage (MSS) and multi-stage (MMS) models, respectively. This is a bit higher than reproted.**

1. Modify scripts

- Set the model path in line81 of `$TracKit/tracking/vot_wrap.py` or `$TracKit/tracking/vot_wrap_mms.py`.

- for model without MMS network (faster): 
```
set running script in vot2020 workspace (i.e. trackers.ini) to `vot_wrap.py`
```
- for model with MMS network (slower):
```
set running script in vot2020 workspace (i.e. trackers.ini) to `vot_wrap_mms.py`
```
- Note: We provided a reference of `trackers.ini` in `$TracKit/trackers.ini`. Please find more running guidelines in VOT official [web](https://www.votchallenge.net/howto/tutorial_python.html).

2. run
```
CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace  $workspace_path OceanPlus
```
- Note: If you only want to test "baseline" track in vot for saving time, please remove lines 10-21 in `$root/anaconda3/envs/TracKit/lib/python3.7/site-packages/vot/stack/vot2020.yaml`.


3. evaluate
```
vot analysis --workspace $workspace_path OceanPlus --output json
```


We also provided the trackers submitted to VOT2020 challenge, i.e. [[OceanPlus]](https://drive.google.com/file/d/1DNDZshPed_fcl1DB2lKiOU1bjYC_dxtp/view?usp=sharing), [[OceanPlus-Online]](https://drive.google.com/file/d/1UahJTVPfV0gcqKlBEFc6nwIaqNhyjKQQ/view?usp=sharing), [[OceanPlus-Online-TRT]](https://drive.google.com/file/d/1pdrgyx6XKzN4b3Cyplnr5bcB4TilRS1y/view?usp=sharing).

#### For VOS
1. prepare data
Download dataset from DAVIS, and then
```
ln -sfb $path_to_DAVIS-trainval $TracKit/dataset/DAVIS
```

2. run
```
CUDA_VISIBLE_DEVICES=0 python tracking/test_oceanplus.py --arch OceanPlus --mms True --dataset DAVIS2016  --resume snapshot/OceanPlusMMS.pth 
```

3. evaluate
```
python lib/core/eval_davis.py --dataset DAVIS2016 --num_threads 1 --datapath dataset/DAVIS
```

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:

The training code will be released after accepted. Thanks for your interest!
 

