#!/bin/bash
# Run ON THE POD after anything works (Gate 3, full run). Captures the exact environment and commands so the run can be
# reproduced without the pod: python/pip freeze, torch/CUDA/driver, GPU model, base image, VideoX-Fun commit + our patch,
# and copies of the scripts as they were actually run. Output: $WORK/env_snapshot_<date>/ ; sync it back to the Mac with
# rsync (see fetch_from_pod.sh).
set -e
WORK=${WORK:-/workspace}; S="$WORK/env_snapshot_$(date +%Y%m%d_%H%M)"; mkdir -p "$S"
{ echo "date: $(date -u)"; echo "host: $(hostname)"; echo "image: ${RUNPOD_POD_IMAGE:-unknown} pod: ${RUNPOD_POD_ID:-unknown} gpu_count: ${RUNPOD_GPU_COUNT:-unknown}";
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null; python -V; pip -V;
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version())"; } > "$S/system.txt" 2>&1
pip freeze > "$S/requirements.lock.txt"
(cd "$WORK/VideoX-Fun" && git rev-parse HEAD > "$S/videox_fun_commit.txt" && git diff > "$S/videox_fun_patch.diff")
cp "$(dirname "$0")"/*.sh "$S/" 2>/dev/null; cp -r "$(dirname "$0")/patched" "$S/" 2>/dev/null
cp "$WORK/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml" "$WORK/VideoX-Fun/config/zero_stage2_config.json" "$S/" 2>/dev/null || true
env | grep -E '^(WORK|META|TOKEN|EPOCHS|STEPS|CUDA_VISIBLE_DEVICES|NCCL_|HF_)' > "$S/env_vars.txt" || true
echo "snapshot written to $S"; ls "$S"
