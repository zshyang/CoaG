#!/bin/bash
# Unseen-scene / unseen-action / hand-authored cases (George, 2026-09-13), one GPU. Usage (on pod): WORK=/workspace ./run_unseen.sh [case ...]
# Scenes: hold-out clips' backgrounds the model never saw (c0249 cathedral nave, c1699 Moscow ballroom, c1549 foggy Tokyo promenade, c0999 olive grove).
# Actions: none of the 118 training action families (tai chi, juggling, walking and chatting, carrying a ladder).
# Authored: control videos drawn in editor/ (never in training), rendered by render_authored.py.
set -e
WORK=${WORK:-/workspace}; R=$WORK/wan_train/run_infer_case.sh; U=$WORK/data/unseen
newest() { ls -t "$1"/*.safetensors 2>/dev/null | grep -v comfyui | head -n 1; }
export LORA_LOW=${LORA_LOW:-$(newest $WORK/out/full_low_t640)}; export LORA_HIGH=${LORA_HIGH:-$(newest $WORK/out/full_high_t640)}
echo "LORA_LOW=$LORA_LOW LORA_HIGH=$LORA_HIGH"
P599_PEOPLE="Two adults hold one spot on the floor: a tall Afro-Cuban man in a sheer black mesh top and wide silver trousers, and a slim Thai woman in a magenta bodysuit, hair slicked."
P599_ACTION="Fully in frame, heels planted, they slice through sharp hand performance, duckwalk a half-step, then spiral into death-drop dips and snap back up, again and again. One continuous shot from a static tripod camera at eye level. Dreamy, glossy, saturated color."
P899_PEOPLE="Three adults cross the open ground on inline skates: a grinning Argentine woman with a blonde ponytail in a pink jacket, a burly Jamaican man in a mustard beanie, and a slim Japanese man in bright green."
P899_ACTION="They dig in from a standstill with hard crossover starts, race away from the camera, then wrench sideways into hard stops, laughing before setting off again in another direction. Crisp, upbeat, candy-bright. One continuous shot from a static tripod at eye level, all three fully framed."
S_CATH="A deconsecrated brick cathedral nave, columns half wrapped in scaffolding and a dry font pushed aside, rigged with cheap colour cans; haze machine fog rolls between the piers while spotlights sweep hard green and red across worn stone flagstones."
S_BALL="Only two work lamps burn in a faded ballroom of a Moscow hotel awaiting renovation, throwing deep shadows over herringbone parquet, a chandelier bagged in plastic overhead, mirrors turned to the wall and draped."
S_PROM="Thick fog swallows a beachfront promenade in Tokyo, its black-and-white mosaic paving curling away in waves, palms reduced to grey ghosts, steam rising from drain vents along the seawall."
S_OLIVE="Sunrise mist threads a small olive grove terrace above a rocky shore, gnarled trunks in rows on dusty pale earth, a dry stone wall at the drop, and heavy surf breaking white on the rocks below."
S_PLAZA="Stamped concrete patterned like slate spreads across a stadium concourse plaza in Patagonia, night floodlights beating down on shuttered ticket gates and a ring of bare planters; dry leaves swirl low over the ground."
S_LAGOON="Sunrise mist lies flat over a frozen lagoon outside Buenos Aires, snow dusting the ice between frozen reed clumps and a leaning wooden duck blind. The air is perfectly calm."
run() { local name=$1; shift; if ls "$WORK/out/samples/$name/"*.mp4 >/dev/null 2>&1; then echo "skip $name"; return; fi; echo "=== $name $(date -u +%H:%M)"; env "$@" TAG=$name $R > "$WORK/out/unseen_$name.log" 2>&1 || echo "FAILED $name"; }
order=(scene_c0599_cathedral scene_c0899_promenade action_c0599_taichi action_c0899_walk_chat authored_two_people_orbit authored_four_people_dolly_in scene_c0599_ballroom scene_c0899_olive action_c0599_juggling action_c0899_ladder)
sel=("$@"); [ $# -eq 0 ] && sel=("${order[@]}")
for c in "${sel[@]}"; do case $c in
  scene_c0599_cathedral) run $c CID=c0599 CAM=orig REF=$U/ref/c0249.png PROMPT="$S_CATH $P599_PEOPLE $P599_ACTION" ;;
  scene_c0599_ballroom)  run $c CID=c0599 CAM=orig REF=$U/ref/c1699.png PROMPT="$S_BALL $P599_PEOPLE $P599_ACTION" ;;
  scene_c0899_promenade) run $c CID=c0899 CAM=orig REF=$U/ref/c1549.png PROMPT="$S_PROM $P899_PEOPLE $P899_ACTION" ;;
  scene_c0899_olive)     run $c CID=c0899 CAM=orig REF=$U/ref/c0999.png PROMPT="$S_OLIVE ${P899_PEOPLE/on inline skates/on foot} ${P899_ACTION/on inline skates/}" ;;
  action_c0599_taichi)   run $c CID=c0599 CAM=orig PROMPT="$S_PLAZA $P599_PEOPLE Fully in frame, they move through a slow tai chi form in unison, weight shifting from foot to foot, arms sweeping in wide circles, never leaving their spots. One continuous shot from a static tripod camera at eye level. Calm, glossy, saturated color." ;;
  action_c0599_juggling) run $c CID=c0599 CAM=orig PROMPT="$S_PLAZA $P599_PEOPLE Fully in frame, each juggles three bright balls, stepping side to side and turning on the spot as the balls arc overhead. One continuous shot from a static tripod camera at eye level. Playful, glossy, saturated color." ;;
  action_c0899_walk_chat) run $c CID=c0899 CAM=orig PROMPT="$S_LAGOON Three adults walk across the ice in a loose line: a grinning Argentine woman with a blonde ponytail in a pink jacket, a burly Jamaican man in a mustard beanie, and a slim Japanese man in bright green. They walk briskly away from the camera, chatting and gesturing with their hands, then spread apart. One continuous shot from a static tripod at eye level, all three fully framed." ;;
  action_c0899_ladder)   run $c CID=c0899 CAM=orig PROMPT="$S_LAGOON Three adults carry a long aluminum ladder together across the ice: a grinning Argentine woman with a blonde ponytail in a pink jacket, a burly Jamaican man in a mustard beanie, and a slim Japanese man in bright green. They shuffle away from the camera holding the ladder at waist height, then swing it around. One continuous shot from a static tripod at eye level, all three fully framed." ;;
  authored_two_people_orbit) run $c CID=c0599 CV=$U/authored/two_people_orbit/control.mp4 REF=$WORK/data/train_data/ref/c0599.png PROMPT="$S_PLAZA Two adults stand on the concrete: a tall Afro-Cuban man in a sheer black mesh top and wide silver trousers, and a slim Thai woman in a magenta bodysuit, hair slicked. They walk past each other, swapping places, then turn to face the camera. One continuous shot, the camera slowly orbiting them at eye level. Dreamy, glossy, saturated color." ;;
  authored_four_people_dolly_in) run $c CID=c0999 CV=$U/authored/four_people_dolly_in/control.mp4 REF=$U/ref/c0999.png PROMPT="$S_OLIVE Four adults stand in a loose row in the dust: a tall Croatian woman with a blonde ponytail in lilac, a broad Black man in white, a compact Filipino woman in gold, and a lean Iranian man in navy. They stretch and shake out their arms, chatting, staying on their spots. One continuous shot, the camera slowly dollying in toward them at eye level." ;;
  *) echo "unknown $c" ;;
esac; done; echo UNSEEN_DONE; ls $WORK/out/samples/*/*.mp4 | wc -l
