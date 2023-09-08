#!/bin/bash
set -euo pipefail

MNT_DIR=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cp "${SCRIPT_DIR}/../../../build/tensorflow-cudart-dynam/prebuilt/tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl" \
    "${MNT_DIR}"
mkdir -p "${MNT_DIR}/cuda_dumps/"
find "${SCRIPT_DIR}/../covidct/cuda_dumps/" -name "*.ava" | xargs -I {} cp {} "${MNT_DIR}/cuda_dumps/"
cp "${SCRIPT_DIR}/../covidct/refactored/zoom_kernel.cubin" "${MNT_DIR}/cuda_dumps/"

model_dir="${MNT_DIR}/models"
mkdir -p ${model_dir}

cp "${SCRIPT_DIR}/../covidct/refactored/models/bcdunet_v2.pb" ${model_dir}
cp "${SCRIPT_DIR}/../covidct/refactored/models/cnn_CovidCtNet_v2_final.pb" ${model_dir}

data_dir="${MNT_DIR}/data"
mkdir -p ${data_dir}
cp ${SCRIPT_DIR}/../covidct/refactored/output/T_normal*.npy ${data_dir}
cp ${SCRIPT_DIR}/../covidct/refactored/input/patient_list.txt ${data_dir}