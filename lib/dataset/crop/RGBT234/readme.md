# Preprocessing RGBT234 (train and val)


### Crop & Generate data info (20 min)

````sh
python RGBT234_genjson.py
python par_crop.py 511 24
python gen_json.py
````
