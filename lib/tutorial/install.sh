#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y -n $conda_env_name  python=3.7 

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib 2.2.2 ******************"
conda install -y matplotlib=2.2.2

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython


echo ""
echo ""
echo "****************** Installing skimage ******************"
pip install scikit-image



echo ""
echo ""
echo "****************** Installing pillow ******************"
pip install 'pillow<7.0.0'

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing shapely ******************"
pip install shapely

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 
pip install mpi4py
pip install ray==0.8.7
pip install hyperopt


echo ""
echo ""
echo "****************** Installing vot python toolkit ******************"
pip install git+https://github.com/votchallenge/vot-toolkit-python@7a1b807df3d64ea310c554e9f487f1e5f53bf249

echo "****************** Installation complete! ******************"
