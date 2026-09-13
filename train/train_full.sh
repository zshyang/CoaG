#!/bin/bash
# Gate 4: full LoRA training, 8 x H100 80GB, DeepSpeed Zero-2. Trains the low-noise expert, then the high-noise expert.
# Usage: WORK=/workspace TOKEN=640 EPOCHS=3 ./train_full.sh [low|high|both]
# TOKEN=640 is the 480p budget (fits 1 GPU each with batch 1); TOKEN=960 is 720p (roughly 3x the time; try after 640 works).
set -e
WORK=${WORK:-/workspace}; cd "$WORK/VideoX-Fun"
export MODEL_NAME="$WORK/models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
export DATASET_NAME="$WORK/data/train_data"
export DATASET_META_NAME="$WORK/data/train_data/${META:-metadata.json}"
TOKEN=${TOKEN:-640}; EPOCHS=${EPOCHS:-3}; WHICH=${1:-both}
run_expert() {
  B=$1
  accelerate launch --num_processes=${NPROC:-8} --num_machines=1 --mixed_precision="bf16" --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard \
    scripts/wan2.2_fun/train_control_lora.py \
    --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_NAME --train_data_meta=$DATASET_META_NAME \
    --image_sample_size=$TOKEN --video_sample_size=$TOKEN --token_sample_size=$TOKEN \
    --video_sample_stride=1 --video_sample_n_frames=81 \
    --train_batch_size=1 --video_repeat=1 --gradient_accumulation_steps=1 --dataloader_num_workers=8 \
    --num_train_epochs=$EPOCHS --checkpointing_steps=200 \
    --learning_rate=1e-04 --seed=42 \
    --output_dir="$WORK/out/full_${B}_t${TOKEN}" \
    --gradient_checkpointing --mixed_precision="bf16" \
    --adam_weight_decay=3e-2 --adam_epsilon=1e-10 --vae_mini_batch=1 --max_grad_norm=0.05 \
    --random_hw_adapt --training_with_video_token_length --enable_bucket --uniform_sampling \
    --train_mode="control_ref" --control_ref_image="random" --add_inpaint_info --add_full_ref_image_in_self_attention \
    --boundary_type="$B" --rank=64 --network_alpha=32 --target_name="q,k,v,ffn.0,ffn.2" --use_peft_lora --low_vram
}
case $WHICH in
  low)  run_expert low ;;
  high) run_expert high ;;
  both) run_expert low; run_expert high ;;
esac
