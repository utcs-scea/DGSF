#!/bin/bash
set -euo pipefail

MNT_DIR=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cp "${SCRIPT_DIR}/../../../build/onnxruntime/prebuilt/onnxruntime_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl" \
    "${MNT_DIR}"
# cp "${SCRIPT_DIR}/../../../build/tensorflow-cudart-dynam/prebuilt/tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl" \
#     "${MNT_DIR}"
# cp "${SCRIPT_DIR}/../../../build/pytorch/prebuilt/torch-1.9.0a0+gitd69c22d-cp36-cp36m-linux_x86_64.whl" \
#     "${MNT_DIR}"
# cp "${SCRIPT_DIR}/../../../build/pytorch/prebuilt/torchvision-0.10.0a0+300a8a4-cp36-cp36m-linux_x86_64.whl" \
#     "${MNT_DIR}"
cp -r "${SCRIPT_DIR}/../mlperf/cuda_dumps/onnxruntime" "${MNT_DIR}/cuda_dumps/"
# cp -r "${SCRIPT_DIR}/../mlperf/cuda_dumps/tensorflow_1_14" "${MNT_DIR}/cuda_dumps/"
# mv "${MNT_DIR}/cuda_dumps/tensorflow_1_14" "${MNT_DIR}/cuda_dumps/tensorflow"
# cp -r "${SCRIPT_DIR}/../mlperf/cuda_dumps/pytorch" "${MNT_DIR}/cuda_dumps/"

model_dir="${MNT_DIR}/models"
mkdir -p ${model_dir}
# onnxruntime model
# find "${SCRIPT_DIR}/../mlperf/models/model_with_external_data/" -name "*.onnx*" | xargs -I {} cp {} ${model_dir}
cp "${SCRIPT_DIR}/../mlperf/models/resnet50_v1.onnx" ${model_dir}
# tensorflow model
# cp "${SCRIPT_DIR}/../mlperf/models/resnet50_v1.pb" ${model_dir}

data_dir="${MNT_DIR}/data"
mkdir -p ${data_dir}/imagenet
cp "${SCRIPT_DIR}/../mlperf/data/ILSVRC2012_img_val/val_map.txt" "${data_dir}/imagenet/val_map.txt"
mkdir -p "${MNT_DIR}/preprocessed/imagenet/NCHW"
rsync -av "${SCRIPT_DIR}/../mlperf/inference/vision/classification_and_detection/preprocessed/imagenet/NCHW/" \
    "${MNT_DIR}/preprocessed/imagenet/NCHW/"
