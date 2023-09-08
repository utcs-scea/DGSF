#!/bin/bash
set -euo pipefail

MNT_DIR=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cp "${SCRIPT_DIR}/../../../build/onnxruntime/prebuilt/onnxruntime_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl" \
    "${MNT_DIR}"
cp -r "${SCRIPT_DIR}/../mlperf/cuda_dumps/onnxruntime" "${MNT_DIR}/cuda_dumps/"

model_dir="${MNT_DIR}/models"
mkdir -p ${model_dir}
# onnxruntime model
cp "${SCRIPT_DIR}/model/updated_withpreprop_r50.onnx" \
    ${model_dir}

data_dir="${MNT_DIR}/data"
mkdir -p ${data_dir}
cp "${SCRIPT_DIR}/../faas_face_det_client/data/wider_val.txt" ${data_dir}/
rsync -av "${SCRIPT_DIR}/../faas_face_det_client/data/WIDER_val/images/" ${data_dir}/