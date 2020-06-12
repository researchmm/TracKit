# SiamDW tutorial
## Testing
### Set up environment
```
bash install.sh $conda_path TrackSeg
```

### Prepare data and models
1. Download the pretrained [PyTorch model](https://drive.google.com/file/d/1SzIql02jJ6Id1k0M6f-zjUA3RgAm6E5U/view?usp=sharing) to `snapshot`.
2. Download [json](https://drive.google.com/open?id=1S-RkzyMVRFWueWW91NmZldUJuDyhGdp1) files of testing data and put thme in `dataset`.
3. Download testing data e.g. VOT2017 and put them in `dataset`. 

### Testing
In root path `$TrackSeg`,
```
python tracking/test_siamdw.py --arch Ocean --resume snapshot/siamdw_res22w.pth --dataset VOT2017
```

We only provide the testing of best model `Res22W` in this repo. If you want to test other models or training, please follow the instructions of [SiamDW](https://github.com/researchmm/SiamDW).
