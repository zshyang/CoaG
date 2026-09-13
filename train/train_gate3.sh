#!/bin/bash
# Gate 3 smoke test: 1 x 80 GB GPU, low-noise expert LoRA, 480p token budget, a few hundred steps.
# Goal: loss goes down, checkpoints save, one inference sample looks controlled. Budget: ~1-2 GPU-hours.
# Usage: WORK=/workspace STEPS=300 ./train_gate3.sh
set -e
WORK=${WORK:-/workspace}; cd "$WORK/VideoX-Fun"
export MODEL_NAME="$WORK/models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
export DATASET_NAME="$WORK/data/train_data"
export DATASET_META_NAME="$WORK/data/train_data/${META:-metadata.json}"
STEPS=${STEPS:-300}
accelerate launch --mixed_precision="bf16" scripts/wan2.2_fun/train_control_lora.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 --video_sample_size=640 --token_sample_size=640 \
  --video_sample_stride=1 --video_sample_n_frames=81 \
  --train_batch_size=1 --video_repeat=1 --gradient_accumulation_steps=1 --dataloader_num_workers=4 \
  --max_train_steps=$STEPS --checkpointing_steps=100 \
  --learning_rate=1e-04 --seed=42 \
  --output_dir="$WORK/out/gate3_low" \
  --gradient_checkpointing --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 --adam_epsilon=1e-10 --vae_mini_batch=1 --max_grad_norm=0.05 \
  --random_hw_adapt --training_with_video_token_length --enable_bucket --uniform_sampling \
  --train_mode="control_ref" --control_ref_image="random" --add_inpaint_info --add_full_ref_image_in_self_attention \
  --boundary_type="low" --rank=64 --network_alpha=32 --target_name="q,k,v,ffn.0,ffn.2" --use_peft_lora --low_vram
