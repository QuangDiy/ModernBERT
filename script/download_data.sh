#!/bin/bash
set -e

DEST_DIR="./data/merged-dataset-1024"

echo "Downloading dataset to $DEST_DIR ..."
echo

mkdir -p "$DEST_DIR"

hf download \
    QuangDuy/merged-dataset-1024 \
    --repo-type dataset \
    --local-dir "$DEST_DIR"

echo
echo "Download completed to $DEST_DIR"
