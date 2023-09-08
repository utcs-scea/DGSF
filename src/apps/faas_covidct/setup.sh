#!/bin/bash
set -euo pipefail
set -x

python3 -m pip install -U keras_preprocessing --no-deps
python3 -m pip install /tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl
rm /tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl