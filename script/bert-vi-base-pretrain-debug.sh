#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Unbuffered Python output - ensures all prints/tracebacks appear immediately
export PYTHONUNBUFFERED=1

# Print each rank's full output, not just the last N lines
export COMPOSER_PRINT_ALL_RANK_OUTPUT=1

# Run with 8 GPUs - redirect all rank outputs to separate log files for easy inspection
LOG_DIR="./debug_logs"
mkdir -p "$LOG_DIR"

composer --nproc 8 main.py yamls/bert/bert-vi-base-pretrain.yaml "$@" 2>&1 | tee "$LOG_DIR/all_ranks.log"

echo ""
echo "=== Searching for errors across all ranks ==="
grep -n "Error\|Traceback\|Exception\|exited with code [^0]" "$LOG_DIR/all_ranks.log" | head -50
