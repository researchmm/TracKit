# Preprocessing COCO

### Download raw images and annotations

````shell
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip ./train2017.zip
unzip ./val2017.zip
unzip ./annotations_trainval2017.zip
cd pycocotools && make && cd ..
````

### Crop & Generate data info (10 min)

````shell
#python par_crop.py [data_path] [crop_size] [num_threads]
python par_crop.py /data/share/coco  511 12   
python gen_json.py
````

Code are modified from SiamMask.
