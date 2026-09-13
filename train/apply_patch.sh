#!/bin/bash
# Copy our patched files into a VideoX-Fun checkout (pinned to commit 968f0e2, 2026-09-04). Usage: ./apply_patch.sh <VideoX-Fun dir>
set -e
R=${1:?VideoX-Fun dir}; H=$(cd "$(dirname "$0")" && pwd)
cp "$H/patched/dataset_image_video.py"  "$R/videox_fun/data/dataset_image_video.py"
cp "$H/patched/train_control_lora.py"   "$R/scripts/wan2.2_fun/train_control_lora.py"
cp "$H/patched/infer_control_ref.py"    "$R/examples/wan2.2_fun/infer_control_ref.py"
echo "patched: external reference image (metadata key ref_file_path) + env-driven inference script"
