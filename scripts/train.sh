#!/bin/bash
set -x

# Args
CONFIG=$1
OUTPUT_DIR=$2

# Preparation
mkdir -p ${OUTPUT_DIR}
RUN_NAME=$(basename ${OUTPUT_DIR})

# Train
llamafactory-cli train ${CONFIG} \
    output_dir=${OUTPUT_DIR} \
    run_name=${RUN_NAME} \
    2>&1 | tee ${OUTPUT_DIR}/train.log

# e.g.
# bash scripts/train.sh examples/qwen3vl/qwen3vl-4b_llava-next_liger.yaml output/qwen3vl-4b_llava-next_liger