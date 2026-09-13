# Run log

One entry per GPU run: date, platform and pod, GPUs, command, metadata file, step time, peak VRAM, result, where the outputs went.

## 2026-09-13 01:28 PDT — Gate 3 smoke (done)

- Platform: RunPod Secure Cloud, EUR-IS-3, pod `jorb6bgaks7yu7` (1 x H100 SXM 80GB, $3.49/h), image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, network volume `jj9ohnr6da` (150 GB STANDARD) at /workspace.
- Data: train_data.tar (9.8 GB) via private HF dataset `zshyang1106/wan-control-demo-train` (upload from the Mac took 35 min with the xet path disabled; direct SSH streams kept resetting), extracted to /workspace/data/train_data (1935 rows).
- Weights: alibaba-pai/Wan2.2-Fun-A14B-Control at /workspace/models/Diffusion_Transformer/ (65 GB on disk).
- Command: `WORK=/workspace STEPS=300 META=metadata.json train_gate3.sh` (token 640, batch 1, low-noise expert, LoRA r64/a32, --low_vram), log /workspace/out/gate3.log, output /workspace/out/gate3_low.
- Measured 01:38: steady ~34-38 s/step (first step 65 s), GPU util ~67% (rest = VAE encodes of target/control/ref/mask + CPU collate), peak VRAM 63 GB, step_loss 0.01-0.09 (flow-matching loss varies by timestep). Far slower than the 8 s/step estimate.
- Budget consequence (balance ~$195): 8 x H100 at 37 s/step = ~2.5 h per epoch per expert -> the planned 3+3 epochs (~$420) does not fit. Revised: 1 epoch low-noise + 1 epoch high-noise (~5 h, ~$140), 81 frames and token 640 kept; a second epoch later if the samples justify it.
- Plan: stop Gate 3 after checkpoint-100 (~02:35), run one inference sample with that LoRA to validate the external-reference path, then launch the 8-GPU run and stop this pod during training.
- Result 02:20: 108 steps, checkpoint-100.safetensors (+ -compatible_with_comfyui) in /workspace/out/gate3_low; sanity_check/*_ref.gif shows the LaMa background (no people) as the reference, so the external-reference patch is active in training. Trainer stopped at 02:22. Inference test with checkpoint-100 on hold-out c0599 (original control, own background, 50 steps, ~10 s/step after warm-up) launched 02:38 -> /workspace/out/samples/gate3_c0599_orig; done 02:52 (50 steps, ~9.5 s/step after warm-up, 12 min incl. load). Result (cloud/gate3/samples/gate3_c0599_orig_sheet.png): background from the reference image is reproduced faithfully (stadium plaza, crowd, lights); the two cylinders are copied almost literally (a red block and a dark textured block at the right positions, with a hint of a person inside the second) — at 100 samples the LoRA has learned position but not yet cylinder -> person. Expected at this step count; the full run sees ~19x more samples per expert.
- 1-GPU pod stopped 02:55 (uptime 2.56 h, ~$9). Pod cost so far ~$5 (Gate 3) + setup time.

## 2026-09-13 02:40 PDT — Gate 4 full run (done 06:40)

- Platform: RunPod Secure Cloud, EUR-IS-3, pod `x1z4mdo1szx92p` "wan-train-8xh100" (8 x H100 SXM 80GB, $27.92/h, 160 vCPU, 1.5 TB RAM), same image, same network volume at /workspace (code, weights, data already there; only pip + patch rerun via bootstrap_pod.sh none).
- Command: `WORK=/workspace EPOCHS=1 TOKEN=640 META=metadata.json train_full.sh both` (low then high; accelerate 8 processes, DeepSpeed Zero-2, batch 1 per GPU, LoRA r64/a32 on q,k,v,ffn.0,ffn.2, --low_vram), log /workspace/out/full.log, outputs /workspace/out/full_low_t640 and full_high_t640, checkpoints every 200 steps + final.
- Measured 02:58: 28 s/step (8 x batch 1), 70 GB per GPU, GPU util 100% -> 241 steps = ~1.9 h per expert, ~3.9 h both, ~$110.
- Expected before launch: 1935 rows / 8 = 242 steps per expert, ~40 s/step -> ~2.7 h per expert, ~5.5 h total, ~$155. The 1-GPU pod is stopped while this runs.
- Result: low-noise expert 241/241 steps in 1 h 45 min (02:55-04:45, checkpoint-200 + checkpoint-241), high-noise expert 241/241 in 1 h 50 min (04:47-06:40 incl. 8 min model load, checkpoint-200 + checkpoint-241); 27.1 s/step both, 70 GB per GPU, no errors. step_loss 0.002-0.13 with isolated one-step spikes to 0.3-0.95 (steps 92/112/133 low, 140/193 high), each followed by a normal step (max_grad_norm 0.05). All four .safetensors (509 MB each) copied to runtime/wan_control_demo/cloud/full_{low,high}_t640/ as they appeared. Env snapshot /workspace/env_snapshot_20260913_1338 (pip freeze, torch 2.8.0+cu128, deepspeed 0.17.0, peft 0.20.0, commit 968f0e2 + patch diff, scripts as run). 8-GPU pod cost for training: 4 h 0 min = ~$112.

## 2026-09-13 06:41 PDT — Comparison experiments (done 07:02)

- The 1-GPU pod could not be restarted ("not enough free GPUs on the host machine"), so the 14 comparison cases run on the still-up 8-GPU pod, 8 at a time (one case per GPU, CUDA_VISIBLE_DEVICES, model_cpu_offload): `run_compare_parallel.sh` -> `run_compare.sh <case>` -> `run_infer_case.sh`. LoRAs: checkpoint-241 low + checkpoint-241 high at weight 0.55, 50 steps, guidance 6, seed 43, 480x832x81. Outputs /workspace/out/samples/<case>/, logs /workspace/out/compare_<case>.log. Expected ~2 rounds x 15 min = ~$15.
- Timing: model load ~4 min with 8 concurrent readers, then 50 steps at ~9 s/step; round 1 (8 cases) 06:41-06:51, round 2 (6 cases) 06:51-07:02. 8-GPU pod stopped 07:04 (total uptime 4 h 23 min). RunPod billing at 07:10: 1-GPU pod $8.98, 8-GPU pod $95.20 (may settle a few dollars higher), total $104.18 + volume. All 14 videos + per-case logs in runtime/wan_control_demo/cloud/samples/ (contact sheets sheet_*.png, viewer compare.html).
- Findings (hold-out clips c0599 two dancers on a stadium plaza, c0899 three skaters on a frozen lagoon; LoRA 0.55/0.55):
  - Baselines: the LoRA now turns cylinders into people. Count, left-to-right order, footprint and height match the cylinders in both clips; the reference image fixes the background (plaza, crowd, lights / lagoon, reeds, duck blind). Motion follows the caption (dance / skate) rather than copying the original choreography.
  - Person swap (prompt only): c0599 -> elderly bearded man in tweed + young woman in a yellow raincoat; c0899 -> red sweater / black puffer with white helmet / blue tracksuit. Layout unchanged, identities changed. Works.
  - Background swap (reference image + scene sentences): dancers on the frozen lagoon (model adds skates), skaters on the plaza (inline skates). Works; layout unchanged.
  - Camera: dolly_in (both clips) strongly followed, subjects grow to a close-up as the cylinders fill the frame; orbit shows the plaza/lagoon from a rotated view by the end; crane gives an elevated view with the crowd rail; pan sweeps the background. dolly_out is only weakly followed (subjects shrink less than the cylinders). Failure case: cam_c0599_pan shows a third person in the first ~20 frames (count error under a moving camera).
  - Artifacts to note for the paper: occasional limb/pose glitches during fast moves, a large snow-spray blob in person_c0899 around frame 60, ponytail smear in bg_c0899_plaza.

