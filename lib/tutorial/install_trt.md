# Install TensorRT
We install TensorRT on RTX2080Ti with CUDA10.0. If you fail to install it, please use pytorch version.

1) install pycuda
```
export C_INCLUDE_PATH=/usr/local/cuda-10.0/include/:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/cuda-10.0/include/:${CPLUS_INCLUDE_PATH}
pip install pycuda
```
2) download tensorrt
- Go to [NVIDIA-TENSORRT](https://developer.nvidia.com/tensorrt) and then  click `Download Now`.
- Login and download TensorRT7 (please select the version that suits for your platform). We use [TensorRT 7.0.0.11 for Ubuntu 18.04 and CUDA 10.0 tar package](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.0/7.0.0.11/tars/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz) in our experiment.

3) install
```bash
tar -zxf TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz
vim ~/.bashrc

# Add codes in your file ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your-TensorRT-lib-path>
# for example
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/zpzhang/TensorRT-7.0.0.11/lib

source ~/.bashrc
conda activate OceanOnlineTRT
cd TensorRT-7.0.0.11/python
# Remember declare your python version=3.7
pip install tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl
cd TensorRT-7.0.0.11/graphsurgeon
pip install graphsurgeon-0.4.1-py2.py3-none-any.whl
```

4) Verify the installation
```
python
import tensorrt
```

5) Install Torch2trt
```
conda activate OceanOnlineTRT
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```
Verify the installation
```
python
import torch2trt
```




### Note
- If you met the error `ImportError: libcudart.so.10.0: cannot open shared object file: No such file or directory`, please run `sudo cp /usr/local/cuda-10.0/lib64/libcudart.so.10.0 /usr/local/lib/libcudart.so.10.0 && sudo ldconfig`.

 - If you met the error `PermissionError: [Errno 13] Permission denied: '/tmp/torch_extensions/_prroi_pooling/lock'`, please remove `/tmp/torch_extensions/ _prroi_pooling` and rerun the tracker. If other user in your machine have compiled prroi pooling before, this may happens. Besides, if you have compiled pproi_pooling before, please remove `/tmp/torch_extensions/`. Otherwise, you may fail to compile in the new conda environment.

