#!/bin/bash
echo Activating python environment
. .venv/bin/activate
echo we are now in the virtual env
echo pip version
pip --version
echo installing needed dependencies
# pip install --upgrade pip
# pip3 install Cython numpy
pip install pandas
pip install torch
pip install segmentation-models-3D
pip install tensorflow==2.16.2 keras==3.3.3
pip install self-attention-cv
pip install matplotlib

export PIP_NO_BUILD_ISOLATION=1
# pip3 install -r requirements.txt
echo running our script
python3 unetr.py
