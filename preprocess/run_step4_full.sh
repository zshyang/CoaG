#!/bin/zsh
# Full step-4 run, detached from the session (the harness kills in-session background jobs under memory pressure).
# Usage: ./run_step4_full.sh <mengyi_full_dir> <out_dir> [workers=2]
# Each worker gets an interleaved shard of ids, skips ids that already have qc.json, appends to step4_report_wN.jsonl.
set -e
MDIR=$1; ODIR=$2; N=${3:-2}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/Users/george_yang/workspace/runtime/diarization/venv/bin/python
CLIP81=/Users/george_yang/workspace/runtime/wan_control_demo/clip81
GC=/Users/george_yang/workspace/runtime/wan_control_demo/geocalib_all.json
mkdir -p "$ODIR"
ALL=($(ls "$MDIR" | grep -E '^c[0-9]{4}$' | sort))
echo "ids: ${#ALL[@]}  workers: $N  out: $ODIR"
for ((w=0; w<N; w++)); do
  SHARD=()
  for ((i=w; i<${#ALL[@]}; i+=N)); do SHARD+=("${ALL[$((i+1))]}"); done
  nohup "$PY" "$HERE/step4_render.py" "$MDIR" "$CLIP81" "$ODIR" --skip-done --tag=w$w --geocalib=$GC "${SHARD[@]}" > "$ODIR/step4_w$w.log" 2>&1 &
  disown
  echo "worker $w pid $! ids ${#SHARD[@]}"
done
