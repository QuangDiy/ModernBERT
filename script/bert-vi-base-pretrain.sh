#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run 
composer --nproc 1 main.py yamls/bert/bert-vi-base-pretrain.yaml "$@"
