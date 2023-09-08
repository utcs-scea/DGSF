#!/bin/bash
set -euo pipefail

apt install -y zlib1g-dev git libpng-dev libjpeg-dev
python3 -m pip install -U pip setuptools
python3 -m pip install opencv-python-headless
# python3 -m pip install -U keras_preprocessing --no-deps
# python3 -m pip install /tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl
# rm /tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl
python3 -m pip install /onnxruntime_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
rm /onnxruntime_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
# CC="cc -mavx2" python3 -m pip install -U --force-reinstall pillow-simd
# python3 -m pip install --prefix=/opt/intel/ipp ipp-devel
# python3 -m pip install git+https://github.com/pytorch/accimage
# python3 -m pip install /torch-1.9.0a0+gitd69c22d-cp36-cp36m-linux_x86_64.whl
# python3 -m pip install /torchvision-0.10.0a0+300a8a4-cp36-cp36m-linux_x86_64.whl
# rm /torch-1.9.0a0+gitd69c22d-cp36-cp36m-linux_x86_64.whl
# rm /torchvision-0.10.0a0+300a8a4-cp36-cp36m-linux_x86_64.whl