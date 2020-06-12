# Preprocessing MSRA10K

### Download raw images and annotations


### Crop & Generate data info (~20 min)

````shell
#python par_crop.py -h
python par_crop.py --enable_mask --num_threads 24
python gen_json.py
````
