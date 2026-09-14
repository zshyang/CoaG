#!/bin/bash
# Re-run of every case that used c0899's reference image, with a bar-free reference (Veo letterboxed c0899; the LaMa background
# inherited 64 px bars and the model copied them). Reference = centre 16:9 crop of the content, rescaled; the c0899 control videos
# are re-rendered with the matching 1.216x zoom (rerender_camera.py ZOOM). Names get the suffix _nb. Usage (on pod): WORK=/workspace ./run_nobars.sh
set -e
WORK=${WORK:-/workspace}; R=$WORK/wan_train/run_infer_case.sh; NB=$WORK/data/nb
newest() { ls -t "$1"/*.safetensors 2>/dev/null | grep -v comfyui | head -n 1; }
export LORA_LOW=${LORA_LOW:-$(newest $WORK/out/full_low_t640)}; export LORA_HIGH=${LORA_HIGH:-$(newest $WORK/out/full_high_t640)}
CAP() { python3 -c "import json; print(json.load(open('$WORK/data/infer_inputs/holdout_captions.json'))['$1'])"; }
P899=$(CAP c0899)
P599_PEOPLE="Two adults hold one spot on the floor: a tall Afro-Cuban man in a sheer black mesh top and wide silver trousers, and a slim Thai woman in a magenta bodysuit, hair slicked."
P599_ACTION="Fully in frame, heels planted, they slice through sharp hand performance, duckwalk a half-step, then spiral into death-drop dips and snap back up, again and again. One continuous shot from a static tripod camera at eye level. Dreamy, glossy, saturated color."
S_LAGOON="Sunrise mist lies flat over a frozen lagoon outside Buenos Aires, snow dusting the ice between frozen reed clumps and a leaning wooden duck blind. The air is perfectly calm."
P899_PERSON=${P899/"a grinning Argentine woman with a blonde ponytail in a pink jacket, a burly Jamaican man in a mustard beanie, and a slim Japanese man in bright green"/"a tall Norwegian man in a red wool sweater, a small Korean woman in a black puffer jacket and white helmet, and a stocky Mexican man in a blue tracksuit"}
P599_BG="Sunrise mist lies flat over a frozen lagoon outside Buenos Aires, snow dusting the ice between frozen reed clumps and a leaning wooden duck blind. The air is perfectly calm. Two adults hold one spot on the ice: a tall Afro-Cuban man in a sheer black mesh top and wide silver trousers, and a slim Thai woman in a magenta bodysuit, hair slicked. Fully in frame, heels planted, they slice through sharp hand performance, duckwalk a half-step, then spiral into death-drop dips and snap back up, again and again. One continuous shot from a static tripod camera at eye level. Dreamy, glossy, saturated color."
REF=$NB/c0899_ref_nobars.png
run() { local name=$1; shift; if ls "$WORK/out/samples/$name/"*.mp4 >/dev/null 2>&1; then echo "skip $name"; return; fi; echo "=== $name $(date -u +%H:%M)"; env "$@" TAG=$name $R > "$WORK/out/nb_$name.log" 2>&1 || echo "FAILED $name"; }
run bg_c0599_lagoon_nb   CID=c0599 CAM=orig REF=$REF PROMPT="$P599_BG"
run base_c0899_nb        CID=c0899 CV=$NB/control_static.mp4 REF=$REF PROMPT="$P899"
run person_c0899_nb      CID=c0899 CV=$NB/control_static.mp4 REF=$REF PROMPT="$P899_PERSON"
run cam_c0899_orbit_nb   CID=c0899 CV=$NB/control_orbit.mp4 REF=$REF PROMPT="$P899"
run cam_c0899_dolly_in_nb CID=c0899 CV=$NB/control_dolly_in.mp4 REF=$REF PROMPT="$P899"
run action_c0899_walk_chat_nb CID=c0899 CV=$NB/control_static.mp4 REF=$REF PROMPT="$S_LAGOON Three adults walk across the ice in a loose line: a grinning Argentine woman with a blonde ponytail in a pink jacket, a burly Jamaican man in a mustard beanie, and a slim Japanese man in bright green. They walk briskly away from the camera, chatting and gesturing with their hands, then spread apart. One continuous shot from a static camera at eye level, all three fully framed."
run action_c0899_ladder_nb CID=c0899 CV=$NB/control_static.mp4 REF=$REF PROMPT="$S_LAGOON Three adults carry a long aluminum ladder together across the ice: a grinning Argentine woman with a blonde ponytail in a pink jacket, a burly Jamaican man in a mustard beanie, and a slim Japanese man in bright green. They shuffle away from the camera holding the ladder at waist height, then swing it around. One continuous shot from a static camera at eye level, all three fully framed."
echo NOBARS_DONE
