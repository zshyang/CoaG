# train/ — LoRA training for Wan2.2-Fun-A14B-Control on the plane-and-cylinder control videos

Read `TRAIN_DESIGN.md` first.
`build_videox_dataset.py` stages `runtime/wan_control_demo/train_data/` from `train_manifest.jsonl` (hard links + metadata.json).
`upstream/` holds the unmodified VideoX-Fun files at commit 968f0e2; `patched/` holds our versions (external reference image, env-driven inference); `apply_patch.sh` copies them into a checkout.
On the pod, in order: `setup_pod.sh` → `train_gate3.sh` → `infer.sh` → `train_full.sh both`.
`pack_train_data.sh` (tar for scp) or `upload_data_hf.sh` (private HF dataset) moves the 9.2 GB training set off the Mac.
`snapshot_env.sh` (on the pod) and `fetch_from_pod.sh` (on the Mac) capture the environment, commands, LoRA checkpoints and samples after each run; `RUNLOG.md` lists every run.
Training uses `metadata.json` (1935 rows); the camera-balanced and no-bystanders variants are optional.
