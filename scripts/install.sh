#!/bin/bash

# 1) Check Python version
python_version=$(python -V 2>&1 | grep -o 'Python [0-9]*.[0-9]*.[0-9]*' | cut -d ' ' -f 2)
if [ "$python_version" != "3.11.0" ]; then
    echo "Warning: Current Python version is not 3.11.0, your version is $python_version."
fi

# 2) Install torch, allowing the user to specify whether it's for cuda117 or cuda118
read -p "Please choose the torch version to install (enter 117 for CUDA 11.7, enter 118 for CUDA 11.8). It is best to choose one of these two versions, or the closest available version if these are not available: " cuda_version
if [ "$cuda_version" == "117" ]; then
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
elif [ "$cuda_version" == "118" ]; then
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
else
    echo "Error: Incorrect input."
    exit 1
fi

# 3) Install xformers==0.0.22
pip install xformers==0.0.22

# 4) Install timm
pip install timm

# 5) Install mmlab related libraries
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.1

directories=("mmsegmentation" "mmdetection")

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        cd $dir
        pip install -e .
        cd ..
    else
        echo "Warning: Directory $dir does not exist. Skipping installation."
    fi
done


# 6) Install selective_scan of Mamba
cd selective_scan
pip install -e .
cd ..