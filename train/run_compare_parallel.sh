#!/bin/bash
# Run the run_compare.sh cases N at a time, one GPU each (for the 8-GPU pod when the 1-GPU pod cannot be started).
# Usage (on pod): WORK=/workspace NGPU=8 ./run_compare_parallel.sh [case ...]   (no args = all cases in run_compare.sh order)
WORK=${WORK:-/workspace}; NGPU=${NGPU:-8}; cd "$WORK"
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(base_c0599 cam_c0599_orbit person_c0599 bg_c0599_lagoon base_c0899 cam_c0599_dolly_in cam_c0599_pan cam_c0599_crane person_c0899 bg_c0899_plaza cam_c0599_static cam_c0599_dolly_out cam_c0899_orbit cam_c0899_dolly_in)
i=0; pids=()
for c in "${cases[@]}"; do
  g=$((i % NGPU))
  echo "$(date -u +%H:%M) launch $c on GPU $g"
  CUDA_VISIBLE_DEVICES=$g "$WORK/wan_train/run_compare.sh" "$c" > "$WORK/out/compare_$c.log" 2>&1 &
  pids+=($!); i=$((i+1))
  if [ $((i % NGPU)) -eq 0 ]; then echo "$(date -u +%H:%M) waiting for round"; wait "${pids[@]}"; pids=(); fi
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
echo "$(date -u +%H:%M) ALL_COMPARE_DONE"; ls "$WORK"/out/samples/*/*.mp4 2>/dev/null
