#!/bin/bash
# One-time setup on a rented GPU box (RunPod / Lambda / Vast, Ubuntu + CUDA 12.x, PyTorch image).
# Usage: WORK=/workspace HF_TOKEN=hf_xxx ./setup_pod.sh
# Result: $WORK/VideoX-Fun (patched), $WORK/models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control (69 GB), $WORK/data/train_data
set -e
WORK=${WORK:-/workspace}; cd "$WORK"
H=$(cd "$(dirname "$0")" && pwd)
# 1. code, pinned
if [ ! -d VideoX-Fun ]; then git clone https://github.com/aigc-apps/VideoX-Fun.git; fi
cd VideoX-Fun && git fetch -q && git checkout -q 968f0e2192 && cd ..
PIP="pip install -q --break-system-packages"          # Runpod PyTorch image marks python as externally managed (PEP 668)
$PIP "huggingface_hub[cli]"
$PIP -r VideoX-Fun/requirements.txt deepspeed==0.17.0 "numpy==1.26.4" peft
export PATH="$PATH:/usr/local/bin:$HOME/.local/bin"; which hf
"$H/apply_patch.sh" "$WORK/VideoX-Fun"
# 2. weights (69 GB; ~10 min on a datacenter link)
mkdir -p models/Diffusion_Transformer data
if [ ! -f models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control/models_t5_umt5-xxl-enc-bf16.pth ]; then
  hf download alibaba-pai/Wan2.2-Fun-A14B-Control --local-dir models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control
fi
du -sh models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control
# 3. data: either the HF private dataset (see upload_data_hf.sh) or a tar copied in with scp/runpodctl
mkdir -p data
if [ -n "$HF_DATA_REPO" ] && [ "$HF_DATA_REPO" != none ] && [ ! -f data/train_data/metadata.json ]; then hf download "$HF_DATA_REPO" train_data.tar --repo-type dataset --local-dir data && tar -xf data/train_data.tar -C data && rm data/train_data.tar; fi
ls data 2>/dev/null | head; [ -f data/train_data/metadata.json ] && python -c "import json; print('metadata rows', len(json.load(open('$WORK/data/train_data/metadata.json'))))" || echo "data not staged yet (scp the tar to $WORK/data/train_data.tar and extract)"; echo SETUP_DONE
