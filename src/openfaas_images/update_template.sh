#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p ${SCRIPT_DIR}/template/python3-dgsf/onnx_opt
cp -r ${SCRIPT_DIR}/../../build/ava/release/onnx_opt/* ${SCRIPT_DIR}/template/python3-dgsf/onnx_opt/

cp -r ${SCRIPT_DIR}/../../tools/ava.conf ${SCRIPT_DIR}/template/python3-dgsf/ava.conf

if [ -d "${SCRIPT_DIR}/../../libs" ]
then
	if [ ! "$(ls -A  ${SCRIPT_DIR}/../../libs)" ]; then
        echo "No libs found in ${SCRIPT_DIR}/../../libs, go in there and download"
        exit 1
	fi
fi

mkdir -p ${SCRIPT_DIR}/template/python3-dgsf/plibs
cp -r ${SCRIPT_DIR}/../../libs/*.whl ${SCRIPT_DIR}/template/python3-dgsf/plibs
