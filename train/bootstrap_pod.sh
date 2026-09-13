#!/bin/bash
# From the Mac: copy the train/ folder and the HF token to a fresh pod and launch setup_pod.sh detached.
# Usage: IP=1.2.3.4 PORT=12345 ./bootstrap_pod.sh [HF_DATA_REPO]
set -e
IP=${IP:?pod public ip}; PORT=${PORT:?pod ssh port}; REPO=${1:-zshyang1106/wan-control-demo-train}
H=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP"
SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT"
$SSH 'mkdir -p /workspace/wan_train /root/.cache/huggingface && hostname && nvidia-smi -L && df -h /workspace | tail -1'
$SCP -r "$H/patched" "$H"/*.sh "$H"/*.py root@$IP:/workspace/wan_train/
$SCP ~/.cache/huggingface/token root@$IP:/root/.cache/huggingface/token
$SSH "chmod +x /workspace/wan_train/*.sh; cd /workspace && export HF_TOKEN=\$(cat /root/.cache/huggingface/token) WORK=/workspace HF_DATA_REPO=$REPO; nohup /workspace/wan_train/setup_pod.sh > /workspace/setup.log 2>&1 < /dev/null & disown; sleep 2; echo 'setup launched'; tail -2 /workspace/setup.log"
