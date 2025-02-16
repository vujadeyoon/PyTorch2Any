#!/bin/bash
#
#
# Developer: vujadeyoon
# Email: vujadeyoon@gmail.com
#
#
readonly path_curr=$(pwd)
readonly path_parents=$(dirname "${path_curr}")
#
#
unset PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED
export PYTHONPATH=$PYTHONPATH:${path_curr}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
#
#
readonly model=${1}
readonly quai_device=${2}
readonly path_dir_onnx=${3-"${path_curr}/ckpt/onnx/"}
readonly path_dir_tflite=${4-"${path_curr}/ckpt/tflite/"}
#
#
python3 "${path_curr}/lib/internal/quaihub/main.py" --model "${1}" \
                                                    --quai_device "${quai_device}" \
                                                    --path_dir_onnx "${path_dir_onnx}" \
                                                    --path_dir_tflite "${path_dir_tflite}"
