#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p ${SCRIPT_DIR}/template/onnx_opt
cp -r ${SCRIPT_DIR}/../../build/ava/release/onnx_opt/* ${SCRIPT_DIR}/template/onnx_opt/
cp ${SCRIPT_DIR}/../../tools/ava.conf ${SCRIPT_DIR}/template/ava.conf

cp ${SCRIPT_DIR}/../../src/apps/covidct/refactored/zoom_kernel.cubin  ${SCRIPT_DIR}/template/zoom_kernel.cubin

cp /usr/local/cuda/bin/cuobjdump ${SCRIPT_DIR}/template

if [ -d "${SCRIPT_DIR}/../../libs" ]
then
	if [ ! "$(ls -A  ${SCRIPT_DIR}/../../libs)" ]; then
        echo "No libs found in ${SCRIPT_DIR}/../../libs, go in there and run the download script"
        exit 1
	fi
fi

mkdir -p ${SCRIPT_DIR}/template/plibs
cp -r ${SCRIPT_DIR}/../../libs/*.whl ${SCRIPT_DIR}/template/plibs
