#!/bin/bash
# Same duel with the camera fitted to the background photo (fit_background.py: GeoCalib focal + horizon), names roman_fit_<path>.
# Usage (on pod): WORK=/workspace ./run_roman.sh [path ...]   (paths: orbit crane dolly_in static; default all four)
set -e
WORK=${WORK:-/workspace}; R=$WORK/wan_train/run_infer_case.sh; U=$WORK/data/unseen
newest() { ls -t "$1"/*.safetensors 2>/dev/null | grep -v comfyui | head -n 1; }
export LORA_LOW=${LORA_LOW:-$(newest $WORK/out/full_low_t640)}; export LORA_HIGH=${LORA_HIGH:-$(newest $WORK/out/full_high_t640)}
SCENE="The sand floor of the Colosseum arena in Rome under a hard midday sun, tiers of ruined stone arcades rising on every side, a low wooden railing at the edge of the sand and pale dust on the ground."
PEOPLE="Two adult men in Roman armor hold the sand: a broad Italian legionary in a red tunic under segmented steel plate, a crested helmet, a short gladius and a tall rectangular shield; and a lean North African fighter in a bronze muscle cuirass and open helmet with a spear and a round shield."
ACTION="Fully in frame, they circle each other warily, close in, clash sword against spear, shield-bash and shove apart, then circle again, swapping sides. Epic cinematic, dusty, high contrast."
declare -A CAM=( [static]="One continuous shot from a static tripod camera at eye level." [orbit]="One continuous shot, the camera slowly orbiting the two fighters at eye level." [crane]="One continuous shot, the camera craning down from high above to eye level." [dolly_in]="One continuous shot, the camera slowly dollying in toward the fighters at eye level." )
sel=("$@"); [ $# -eq 0 ] && sel=(orbit dolly_in pan static)
for p in "${sel[@]}"; do
  name=roman_fit_$p; if ls "$WORK/out/samples/$name/"*.mp4 >/dev/null 2>&1; then echo "skip $name"; continue; fi
  echo "=== $name $(date -u +%H:%M)"
  CID=c0599 CV=$U/authored/roman_duel_fit_$p/control.mp4 REF=$U/ref/colosseum.png PROMPT="$SCENE $PEOPLE $ACTION ${CAM[$p]}" TAG=$name $R > "$WORK/out/roman_$name.log" 2>&1 || echo "FAILED $name"
done; echo ROMAN_FIT_DONE
