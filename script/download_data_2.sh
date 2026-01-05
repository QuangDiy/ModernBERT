#!/bin/bash
set -e

DEST_DIR="/workspace/ModernBERT/data/merged-dataset-4096"

echo "Downloading dataset to $DEST_DIR ..."
echo

mkdir -p "$DEST_DIR" "$TMPDIR"

python - << 'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="QuangDuy/merged-chunked-4096-v2",
    repo_type="dataset",
    local_dir="/workspace/ModernBERT/data/merged-dataset-4096",
    local_dir_use_symlinks=False,
    resume_download=True,
)
EOF

echo
echo "Download completed to $DEST_DIR"
