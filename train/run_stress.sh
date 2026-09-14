#!/bin/bash
# Stress tests likely to fail (George, 2026-09-13: "find videos that might break"), one GPU. Background: hold-out c0599 plaza.
# Usage (on pod): WORK=/workspace ./run_stress.sh [case ...]
set -e
WORK=${WORK:-/workspace}; R=$WORK/wan_train/run_infer_case.sh; U=$WORK/data/unseen/authored/stress
newest() { ls -t "$1"/*.safetensors 2>/dev/null | grep -v comfyui | head -n 1; }
export LORA_LOW=${LORA_LOW:-$(newest $WORK/out/full_low_t640)}; export LORA_HIGH=${LORA_HIGH:-$(newest $WORK/out/full_high_t640)}
S="Stamped concrete patterned like slate spreads across a stadium concourse plaza in Patagonia, night floodlights beating down on shuttered ticket gates and a ring of bare planters; dry leaves swirl low over the ground."
E="Dreamy, glossy, saturated color."
declare -A P
P[stress_6people]="Six adults in casual streetwear stand in a loose diagonal line across the plaza, chatting, shifting their weight and gesturing. One continuous shot from a static tripod camera at eye level. $E"
P[stress_8people]="Eight adults in casual streetwear stand in two loose rows on the plaza, chatting and gesturing. One continuous shot from a static camera at eye level. $E"
P[stress_cross_twice]="Two adults, a man in a grey hoodie and a woman in a red coat, walk back and forth across the plaza, passing each other twice. One continuous shot from a static camera at eye level. $E"
P[stress_orbit90]="Two adults hold one spot of concrete: a tall Afro-Cuban man in a sheer black mesh top and wide silver trousers, and a slim Thai woman in a magenta bodysuit. They dance in place with sharp arm work. One continuous shot, the camera orbiting a quarter turn around them at eye level. $E"
P[stress_dolly_far]="Two adults hold one spot of concrete: a tall Afro-Cuban man in a sheer black mesh top and wide silver trousers, and a slim Thai woman in a magenta bodysuit. They dance in place. One continuous shot, the camera dollying in fast from far away until the two fill the frame. $E"
P[stress_topdown]="Two adults, a man in silver trousers and a woman in a magenta bodysuit, dance in place on the plaza. One continuous shot from a camera high above, looking steeply down at them. $E"
P[stress_lowcam]="Two adults, a man in silver trousers and a woman in a magenta bodysuit, dance in place on the plaza. One continuous shot from a camera lying on the ground, looking up at them. $E"
P[stress_tall_short]="A very tall man in a long black coat and a small child in a yellow raincoat stand side by side on the plaza, holding hands and swaying. One continuous shot from a static camera at eye level. $E"
P[stress_contradict_count]="One woman in a magenta bodysuit stands alone on the empty plaza and dances in place. Nobody else is present. One continuous shot from a static camera at eye level. $E"
P[stress_seated]="Two adults sit cross-legged on the concrete facing the camera, talking and laughing, never standing up. One continuous shot from a static camera at eye level. $E"
P[stress_exit_frame]="A man in a grey hoodie walks from far away straight toward the camera and past it, while a woman in a red coat waits to the side. One continuous shot from a static camera at eye level. $E"
P[stress_empty]="The plaza is completely empty, nobody present, only dry leaves blowing across the concrete under the floodlights. One continuous shot from a static camera at eye level. $E"
order=(stress_6people stress_cross_twice stress_orbit90 stress_tall_short stress_contradict_count stress_seated stress_exit_frame stress_topdown stress_8people stress_lowcam stress_dolly_far stress_empty)
sel=("$@"); [ $# -eq 0 ] && sel=("${order[@]}")
for c in "${sel[@]}"; do
  if ls "$WORK/out/samples/$c/"*.mp4 >/dev/null 2>&1; then echo "skip $c"; continue; fi
  echo "=== $c $(date -u +%H:%M)"
  CID=c0599 CV=$U/$c/control.mp4 REF=$WORK/data/train_data/ref/c0599.png PROMPT="$S ${P[$c]}" TAG=$c $R > "$WORK/out/stress_$c.log" 2>&1 || echo "FAILED $c"
done; echo STRESS_DONE
