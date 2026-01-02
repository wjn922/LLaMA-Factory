#!/bin/bash
set -x

MODEL=$1
MODEL_ARGS=$2
TASKS=$3
TASKS_SUFFIX="${TASKS//,/_}"
GPUS=${GPUS:-8}

accelerate launch \
--num_processes=${GPUS} --main_process_port 12399 -m lmms_eval \
--model=${MODEL} \
--model_args=${MODEL_ARGS} \
--tasks=${TASKS} \
--batch_size=1 \
--log_samples \
--log_samples_suffix=${TASKS_SUFFIX} \
--output_path=./output_eval/

# e.g. 
# bash scripts/eval.sh qwen3_vl pretrained=pretrained_models/Qwen3-VL-4B-Instruct,attn_implementation=flash_attention_2,max_pixels=3240000 mme