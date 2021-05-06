# SiamDW tutorial
## Testing

We assume the root path is $TracKit, e.g. `/home/zpzhang/TracKit`
### Set up environment
Please follow [readme of Ocean](../Ocean/ocean.md) to install the environment.

### Prepare data and models
1. Download the pretrained [PyTorch model](https://drive.google.com/file/d/1SzIql02jJ6Id1k0M6f-zjUA3RgAm6E5U/view?usp=sharing) to `$TracKit/snapshot`.
2. Download [json](https://drive.google.com/open?id=1S-RkzyMVRFWueWW91NmZldUJuDyhGdp1) files of testing data and put thme in `$TracKit/dataset`.
3. Download testing data e.g. VOT2017 and put them in `$TracKit/dataset`. 

### Testing
In root path `$TracKit`,
```
python tracking/test_siamdw.py --arch SiamDW --resume snapshot/siamdw_res22w.pth --dataset VOT2017
```


### Training
In root path `$TracKit`,
1. Download pretrain model from [here](https://drive.google.com/file/d/1wXyW82idctCd4FkqKxvuWsL707joEIeI/view?usp=sharing) and put it in `pretrain` (named with `pretrain.model`).

2. modify `experiments/train/SiamDW.yaml` according to your needs. (pls use GOT10K with 20w pairs each epoch in my opinion)
```
python tracking/train_siamdw.py
```

Then, pls follow the `epoch testing` and `tuning` as in Ocean.
