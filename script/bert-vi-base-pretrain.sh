#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Increase NCCL timeout from default 10 minutes to 1 hours 
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# Run 
composer --nproc 8 main.py yamls/bert/bert-vi-base-pretrain.yaml "$@"
