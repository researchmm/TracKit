#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

#conda_install_path=$1
conda_env_name=$1

#source $conda_install_path/etc/profile.d/conda.sh

echo "****************** Installing packages for AdaFree tracker******************"
echo ""
echo ""
echo "****************** Create conda environment ******************"
conda create -y --name $conda_env_name python=3.6
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing Cython ******************"
pip install cython --user

echo ""
echo ""
echo "****************** Installing pytorch 1.1.0 ******************"
pip install torch==1.1.0 --user

echo ""
echo ""
echo "****************** Installing torchvision 0.2.1 ******************"
pip install torchvision==0.2.1 --user

echo ""
echo ""
echo "****************** Installing numba/colorma ******************"
pip install numba --user
pip install colorma --user

echo ""
echo ""
echo "****************** Installing opencv-python 4.0.0.21 ******************"
pip install opencv-python==4.0.0.21 --user

echo ""
echo ""
echo "****************** Installing easydict/shapely ******************"
pip install easydict --user
pip install shapely --user

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX==1.6 --user


echo ""
echo ""
echo "****************** Installing mpi4py ******************"
pip install mpi4py --user


echo ""
echo ""
echo "****************** Installing ray/hyperopt ******************"
pip install ray==0.6.3 --user
pip install hyperopt==0.1.2 --user

echo ""
echo ""
echo "****************** Installing DCN conv ******************"
python setup.py develop
