#!/bin/bash
# Pull LoRA checkpoints, logs, samples and env snapshots from the pod to the Mac (and mirror to the external drive).
# Usage: POD="root@<ip> -p <port>" ./fetch_from_pod.sh      (RunPod shows the ssh command on the pod page)
set -e
POD=${POD:?ssh target, e.g. 'root@1.2.3.4 -p 12345'}
DST=/Users/george_yang/workspace/runtime/wan_control_demo/cloud; mkdir -p "$DST"
HOST=${POD%% *}; PORT=$(echo "$POD" | sed -n 's/.*-p \([0-9]*\).*/\1/p'); PORT=${PORT:-22}
rsync -avz --progress -e "ssh -p $PORT" --include='*/' --include='*.safetensors' --include='*.log' --include='*.txt' --include='*.json' --include='*.diff' --include='*.yaml' --include='*.mp4' --include='*.sh' --include='*.py' --exclude='*' \
  "$HOST:/workspace/out/" "$DST/out/"
rsync -avz --progress -e "ssh -p $PORT" "$HOST:/workspace/env_snapshot_*" "$DST/" 2>/dev/null || true
rsync -avz --progress -e "ssh -p $PORT" "$HOST:/workspace/VideoX-Fun/samples/" "$DST/samples/" 2>/dev/null || true
if [ -d /Volumes/ATLAS_4T/wan_control_demo ]; then rsync -a "$DST/" /Volumes/ATLAS_4T/wan_control_demo/cloud/ && echo "mirrored to ATLAS_4T"; fi
du -sh "$DST"
