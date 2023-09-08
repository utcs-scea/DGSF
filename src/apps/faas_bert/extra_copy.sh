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
cp "${SCRIPT_DIR}/../mlperf/inference/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx" \
    ${model_dir}

data_dir="${MNT_DIR}/data"
mkdir -p ${data_dir}
cp "${SCRIPT_DIR}/../mlperf/inference/language/bert/eval_features.pickle" \
    ${data_dir}