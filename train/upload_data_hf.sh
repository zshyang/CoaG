#!/bin/bash
# Optional: push the training set to a PRIVATE Hugging Face dataset repo so any pod can pull it at datacenter speed.
# Usage: HF_TOKEN=hf_xxx REPO=zshyang/wan-control-demo-train ./upload_data_hf.sh
set -e
REPO=${REPO:?repo id like user/name}
hf repo create "$REPO" --repo-type dataset --private || true
hf upload "$REPO" /Users/george_yang/workspace/runtime/wan_control_demo/train_data . --repo-type dataset --commit-message "wan control demo train set (v2 step-4, count_mismatch soft)"
echo "on the pod: HF_DATA_REPO=$REPO ./setup_pod.sh"
