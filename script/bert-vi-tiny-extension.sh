#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run 
composer --nproc 4 main.py yamls/bert/bert-vi-tiny-context-extension.yaml "$@"
