#!/bin/bash
# Generate one clip from a control video + reference image + prompt with the trained LoRA(s).
# Usage: WORK=/workspace LORA_LOW=out/full_low_t640/checkpoint-XXXX.safetensors LORA_HIGH=... \
#        CONTROL_VIDEO=path.mp4 REF_IMAGE=bg.png PROMPT="..." SAMPLE_SIZE=480,832 SAVE_PATH=samples/x ./infer.sh
set -e
WORK=${WORK:-/workspace}; cd "$WORK/VideoX-Fun"
export MODEL_NAME=${MODEL_NAME:-$WORK/models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control}
python examples/wan2.2_fun/infer_control_ref.py
