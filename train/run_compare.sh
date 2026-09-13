#!/bin/bash
# Comparison experiments on one GPU after Gate 4 (George, 2026-09-13): (1) change the people via the prompt,
# (2) change the background via another clip's reference image, (3) change the camera via re-rendered control videos.
# Usage (on pod): WORK=/workspace ./run_compare.sh [case ...]   (no args = all cases, in priority order)
# LoRAs default to the newest checkpoint of full_low_t640 / full_high_t640. Each case ~10 min (load + 50 steps, 480x832x81).
set -e
WORK=${WORK:-/workspace}; R=$WORK/wan_train/run_infer_case.sh
newest() { ls -t "$1"/checkpoint-*.safetensors 2>/dev/null | grep -v comfyui | head -n 1; }
export LORA_LOW=${LORA_LOW:-$(newest $WORK/out/full_low_t640)}
export LORA_HIGH=${LORA_HIGH:-$(newest $WORK/out/full_high_t640)}
echo "LORA_LOW=$LORA_LOW LORA_HIGH=$LORA_HIGH"
CAP() { python3 -c "import json; print(json.load(open('$WORK/data/infer_inputs/holdout_captions.json'))['$1'])"; }
P599=$(CAP c0599); P899=$(CAP c0899)
# (1) people swapped, everything else identical
P599_PERSON=${P599/"a tall Afro-Cuban man in a sheer black mesh top and wide silver trousers, and a slim Thai woman in a magenta bodysuit, hair slicked"/"an elderly Scottish man with a white beard in a brown tweed suit, and a young Nigerian woman in a bright yellow raincoat, hair in long braids"}
P899_PERSON=${P899/"a grinning Argentine woman with a blonde ponytail in a pink jacket, a burly Jamaican man in a mustard beanie, and a slim Japanese man in bright green"/"a tall Norwegian man in a red wool sweater, a small Korean woman in a black puffer jacket and white helmet, and a stocky Mexican man in a blue tracksuit"}
# (2) background swapped: reference image from the other clip, scene sentences rewritten to match, people + action + camera kept
P599_BG="Sunrise mist lies flat over a frozen lagoon outside Buenos Aires, snow dusting the ice between frozen reed clumps and a leaning wooden duck blind. The air is perfectly calm. Two adults hold one spot on the ice: a tall Afro-Cuban man in a sheer black mesh top and wide silver trousers, and a slim Thai woman in a magenta bodysuit, hair slicked. Fully in frame, heels planted, they slice through sharp hand performance, duckwalk a half-step, then spiral into death-drop dips and snap back up, again and again. One continuous shot from a static tripod camera at eye level. Dreamy, glossy, saturated color."
P899_BG="Stamped concrete patterned like slate spreads across a stadium concourse plaza in Patagonia, night floodlights beating down on shuttered ticket gates and a ring of bare planters; dry leaves swirl low over the ground. Three adults on inline skates cross the open plaza: a grinning Argentine woman with a blonde ponytail in a pink jacket, a burly Jamaican man in a mustard beanie, and a slim Japanese man in bright green. They dig in from a standstill with hard crossover starts, race away toward the railing, then wrench sideways into hard stops, laughing before setting off again in another direction. Crisp, upbeat, candy-bright. One continuous shot from a static tripod at eye level, all three fully framed."
run() { # name cid cam prompt [ref]
  local name=$1 cid=$2 cam=$3 prompt=$4 ref=$5
  if [ -f "$WORK/out/samples/$name/"*.mp4 ] 2>/dev/null || ls "$WORK/out/samples/$name/"*.mp4 >/dev/null 2>&1; then echo "skip $name (done)"; return; fi
  echo "=== $name  $(date -u +%H:%M)"
  CID=$cid CAM=$cam TAG=$name PROMPT="$prompt" REF=${ref:-} $R > "$WORK/out/samples_$name.log" 2>&1 || echo "FAILED $name (see samples_$name.log)"
}
declare -A CASES
order=(base_c0599 cam_c0599_orbit person_c0599 bg_c0599_lagoon base_c0899 cam_c0599_dolly_in cam_c0599_pan cam_c0599_crane person_c0899 bg_c0899_plaza cam_c0599_static cam_c0599_dolly_out cam_c0899_orbit cam_c0899_dolly_in)
sel=("${@:-${order[@]}}"); [ $# -eq 0 ] && sel=("${order[@]}")
for c in "${sel[@]}"; do case $c in
  base_c0599)        run $c c0599 orig "$P599" ;;
  base_c0899)        run $c c0899 orig "$P899" ;;
  person_c0599)      run $c c0599 orig "$P599_PERSON" ;;
  person_c0899)      run $c c0899 orig "$P899_PERSON" ;;
  bg_c0599_lagoon)   run $c c0599 orig "$P599_BG" $WORK/data/train_data/ref/c0899.png ;;
  bg_c0899_plaza)    run $c c0899 orig "$P899_BG" $WORK/data/train_data/ref/c0599.png ;;
  cam_c0599_*)       run $c c0599 ${c#cam_c0599_} "$P599" ;;
  cam_c0899_*)       run $c c0899 ${c#cam_c0899_} "$P899" ;;
  *) echo "unknown case $c" ;;
esac; done
echo COMPARE_DONE
