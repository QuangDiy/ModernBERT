#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run 
composer --nproc 8 main.py yamls/bert/bert-vi-base-pretrain.yaml "$@"
