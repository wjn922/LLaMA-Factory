#!/bin/bash
set -x

# Args
CONFIG=$1
OUTPUT_DIR=$2
FSDP2_CONFIG=${FSDP2_CONFIG:-examples/accelerate/fsdp2_config.yaml}

# Preparation
mkdir -p ${OUTPUT_DIR}
RUN_NAME=$(basename ${OUTPUT_DIR})

# Train
accelerate launch \
    --config_file ${FSDP2_CONFIG} \
    src/train.py ${CONFIG} \
    output_dir=${OUTPUT_DIR} \
    run_name=${RUN_NAME} \
    2>&1 | tee ${OUTPUT_DIR}/train.log

# e.g.
# bash scripts/train_fsdp2.sh examples/qwen3vl/qwen3vl-4b_llava-next_liger.yaml output/qwen3vl-4b_llava-next_liger