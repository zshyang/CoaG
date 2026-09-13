#!/bin/bash
# One inference case on the pod. Usage (on pod):
#   CID=c0599 CAM=static LORA_LOW=/workspace/out/.../low.safetensors [LORA_HIGH=...] [PROMPT="..."] [REF=/path.png] [TAG=name] ./run_infer_case.sh
# Defaults: control = /workspace/data/infer_inputs/$CID/control_$CAM.mp4 (CAM=orig -> the training control video),
#           ref = /workspace/data/train_data/ref/$CID.png, prompt = the clip's caption (holdout_captions.json), 480x832, 81 frames.
set -e
WORK=${WORK:-/workspace}; CID=${CID:?}; CAM=${CAM:-orig}; TAG=${TAG:-$CID_$CAM}
if [ "$CAM" = orig ]; then CV=$WORK/data/train_data/control/$CID.mp4; else CV=$WORK/data/infer_inputs/$CID/control_$CAM.mp4; fi
export CONTROL_VIDEO=$CV
export REF_IMAGE=${REF:-$WORK/data/train_data/ref/$CID.png}
export PROMPT=${PROMPT:-$(python3 -c "import json,sys; print(json.load(open('$WORK/data/infer_inputs/holdout_captions.json'))['$CID'])")}
export SAMPLE_SIZE=${SAMPLE_SIZE:-480,832}
export SAVE_PATH=$WORK/out/samples/$TAG
export GPU_MEMORY_MODE=${GPU_MEMORY_MODE:-model_cpu_offload}
export MODEL_NAME=$WORK/models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control
mkdir -p "$SAVE_PATH"; echo "$PROMPT" > "$SAVE_PATH/prompt.txt"; echo "control=$CONTROL_VIDEO ref=$REF_IMAGE lora_low=$LORA_LOW lora_high=$LORA_HIGH" > "$SAVE_PATH/inputs.txt"
cd $WORK/VideoX-Fun && python examples/wan2.2_fun/infer_control_ref.py
ls -la "$SAVE_PATH"
