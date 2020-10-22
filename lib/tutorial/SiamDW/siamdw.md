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

We only provide the testing of best model `Res22W` in this repo. If you want to test other models or training, please follow the instructions of [SiamDW](https://github.com/researchmm/SiamDW). The testing hype-parameters and training of Res22W will be updated later.
