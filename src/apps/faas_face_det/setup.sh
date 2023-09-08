#!/bin/bash
set -euo pipefail
apt install -y libpng-dev libjpeg-dev
python3 -m pip install -U pip setuptools
python3 -m pip install opencv-python-headless
python3 -m pip install /onnxruntime_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
rm /onnxruntime_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl