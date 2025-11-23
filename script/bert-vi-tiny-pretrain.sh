#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Run 
composer --nproc 1 main.py yamls/bert/bert-vi-tiny-pretrain.yaml "$@"
