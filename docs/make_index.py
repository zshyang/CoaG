"""Build docs/index.html (the project page) from the sample folders in runtime/wan_control_demo/cloud/.
Copies the needed videos/images into docs/assets/ and writes index.html. Re-run whenever new samples land."""
import html, os, shutil
from pathlib import Path
RT = Path('/Users/george_yang/workspace/runtime/wan_control_demo'); HERE = Path(__file__).resolve().parent; A = HERE / 'assets'
(A / 'samples').mkdir(parents=True, exist_ok=True); (A / 'figures').mkdir(parents=True, exist_ok=True)
def put(src, name):
    src = Path(src); dst = A / 'samples' / name
    if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size): shutil.copy2(src, dst)
    return dst.exists()
def vid(name, cap): return f'<figure><video src="assets/samples/{name}" muted loop autoplay playsinline preload="metadata"></video><figcaption>{cap}</figcaption></figure>'
def img(name, cap): return f'<figure><img src="assets/samples/{name}" loading="lazy"><figcaption>{cap}</figcaption></figure>'
# ---- sections: (title, intro, [ (desc, control_src, gen_src, ref_src or None, tag) ])
S1 = RT / 'cloud/samples'; S2 = RT / 'cloud/samples_unseen'; S3 = RT / 'cloud/samples_roman'
C = lambda cid: RT / f'train_data/control/{cid}.mp4'; R = lambda cid: RT / f'train_data/ref/{cid}.png'; I = lambda cid, m: RT / f'infer_inputs/{cid}/control_{m}.mp4'
sections = [
 ('Hand-authored geometry, never seen in training', 'Control videos drawn in the editor (top view, Bézier paths, camera presets) and rendered with the same renderer as the training data.',
  [('Two Roman soldiers fight in the Colosseum (background: a Wikimedia Commons photo, CC BY-SA 2.0 daryl_mitchell, tourists patched out). Camera fitted to the photograph with GeoCalib (focal length + horizon), static.', S3 / 'control_roman_fit_static.mp4', S3 / 'roman_fit_static/00000003.mp4', S3 / 'colosseum_ref.png', 'roman_fit_static'),
   ('Fitted camera, orbit anchored to the photo pose.', S3 / 'control_roman_fit_orbit.mp4', S3 / 'roman_fit_orbit/00000003.mp4', None, 'roman_fit_orbit'),
   ('Fitted camera, dolly in.', S3 / 'control_roman_fit_dolly_in.mp4', S3 / 'roman_fit_dolly_in/00000003.mp4', None, 'roman_fit_dolly_in'),
   ('Fitted camera, pan (failure: a helmet fills the left of the first frames).', S3 / 'control_roman_fit_pan.mp4', S3 / 'roman_fit_pan/00000003.mp4', None, 'roman_fit_pan'),
   ('Same duel WITHOUT fitting the ground (default authoring camera): the fighters come out far larger than the cylinders prescribe. Static.', S3 / 'control_roman_static.mp4', S3 / 'roman_static/00000003.mp4', None, 'roman_static'),
   ('Default camera, orbit.', S3 / 'control_roman_orbit.mp4', S3 / 'roman_orbit/00000003.mp4', None, 'roman_orbit'),
   ('Default camera, dolly in.', S3 / 'control_roman_dolly_in.mp4', S3 / 'roman_dolly_in/00000003.mp4', None, 'roman_dolly_in'),
   ('Default camera, crane (not followed: the video stays at the photo viewpoint).', S3 / 'control_roman_crane.mp4', S3 / 'roman_crane/00000003.mp4', None, 'roman_crane'),
   ('Two people swap places along curved paths under an orbiting camera (stadium plaza background of hold-out c0599).', S2 / 'control_authored_two_people_orbit.mp4', S2 / 'authored_two_people_orbit/00000003.mp4', R('c0599'), 'authored_two_people_orbit'),
   ('Four people of different heights in a row, dolly in (olive grove background of hold-out c0999).', S2 / 'control_authored_four_people_dolly_in.mp4', S2 / 'authored_four_people_dolly_in/00000003.mp4', R('c0999'), 'authored_four_people_dolly_in')]),
 ('Unseen scenes: the background reference image comes from a different hold-out clip', 'Geometry of one hold-out clip, background of another; the model saw neither.',
  [('Two dancers (c0599 geometry) in a deconsecrated cathedral (reference of c0249).', C('c0599'), S2 / 'scene_c0599_cathedral/00000003.mp4', R('c0249'), 'scene_c0599_cathedral'),
   ('Two dancers in a faded Moscow ballroom (reference of c1699).', C('c0599'), S2 / 'scene_c0599_ballroom/00000003.mp4', R('c1699'), 'scene_c0599_ballroom'),
   ('Three skaters (c0899 geometry) on a foggy Tokyo promenade (reference of c1549).', C('c0899'), S2 / 'scene_c0899_promenade/00000003.mp4', R('c1549'), 'scene_c0899_promenade'),
   ('Three people on an olive grove terrace (reference of c0999).', C('c0899'), S2 / 'scene_c0899_olive/00000003.mp4', R('c0999'), 'scene_c0899_olive')]),
 ('Unseen actions: prompts outside the 118 training action families', 'Same geometry and background as the baselines, only the action in the text changed.',
  [('Slow tai chi in unison.', C('c0599'), S2 / 'action_c0599_taichi/00000003.mp4', R('c0599'), 'action_c0599_taichi'),
   ('Juggling three balls each.', C('c0599'), S2 / 'action_c0599_juggling/00000003.mp4', R('c0599'), 'action_c0599_juggling'),
   ('Walking briskly and chatting.', C('c0899'), S2 / 'action_c0899_walk_chat/00000003.mp4', R('c0899'), 'action_c0899_walk_chat'),
   ('Carrying a long ladder together.', C('c0899'), S2 / 'action_c0899_ladder/00000003.mp4', R('c0899'), 'action_c0899_ladder')]),
 ('Stress tests: cases chosen to break the model', 'Hand-authored geometry on the plaza background; each pushes one factor outside the training range (count, occlusion, camera, viewpoint, height ratio, or a prompt that contradicts the geometry).',
  [(d, RT / f'../../code/wan_control_demo/editor/examples/out/stress/{t}/{t}/control.mp4', RT / f'cloud/samples_stress/{t}/00000003.mp4', None, t) for t, d in [
   ('stress_6people', 'Six people in a diagonal line (the training maximum).'), ('stress_8people', 'Eight people (beyond the training range; colours cycle).'),
   ('stress_cross_twice', 'Two people crossing paths twice (occlusion).'), ('stress_orbit90', 'Orbit of 90 degrees (training: +-20).'), ('stress_dolly_far', 'Dolly in over 2.4 subject heights (training: at most 1.2).'),
   ('stress_topdown', 'Camera 3 subject heights up, pitched 55 degrees down.'), ('stress_lowcam', 'Camera 0.2 subject heights above the ground, level.'), ('stress_tall_short', 'A 1.7-unit adult next to a 0.6-unit child.'),
   ('stress_contradict_count', 'Three cylinders, but the prompt says one woman alone.'), ('stress_seated', 'Standing cylinders, but the prompt says two people sit on the ground.'), ('stress_exit_frame', 'One person walks from far away past the camera.'), ('stress_empty', 'No cylinders at all; the prompt asks for an empty plaza.')]]),
 ('Baselines on hold-out clips', 'Own caption, own background, the control video recovered from the clip; the original Veo clip for reference.',
  [('c0599: two dancers on a stadium plaza.', C('c0599'), S1 / 'base_c0599/00000003.mp4', RT / 'train_data/videos/c0599.mp4', 'base_c0599'),
   ('c0899: three skaters on a frozen lagoon.', C('c0899'), S1 / 'base_c0899/00000003.mp4', RT / 'train_data/videos/c0899.mp4', 'base_c0899')]),
 ('Change the people (text only)', 'Same control video and background as the baseline.',
  [('c0599: elderly Scottish man in tweed + young Nigerian woman in a yellow raincoat.', C('c0599'), S1 / 'person_c0599/00000003.mp4', None, 'person_c0599'),
   ('c0899: Norwegian man in a red sweater, Korean woman in a black puffer and white helmet, Mexican man in a blue tracksuit.', C('c0899'), S1 / 'person_c0899/00000003.mp4', None, 'person_c0899')]),
 ('Change the background (reference image)', 'Same control video; the reference image and the scene words swapped between the two clips.',
  [('c0599 dancers on the frozen lagoon.', C('c0599'), S1 / 'bg_c0599_lagoon/00000003.mp4', R('c0899'), 'bg_c0599_lagoon'),
   ('c0899 skaters on the stadium plaza.', C('c0899'), S1 / 'bg_c0899_plaza/00000003.mp4', R('c0599'), 'bg_c0899_plaza')]),
 ('Change the camera (authored paths on the recovered geometry)', 'Same cylinders, caption and reference image; the camera path re-rendered.',
  [(f'c0599: {m.replace("_", " ")}.', I('c0599', m), S1 / f'cam_c0599_{m}/00000003.mp4', None, f'cam_c0599_{m}') for m in ['static', 'dolly_in', 'dolly_out', 'orbit', 'pan', 'crane']] +
  [(f'c0899: {m.replace("_", " ")}.', I('c0899', m), S1 / f'cam_c0899_{m}/00000003.mp4', None, f'cam_c0899_{m}') for m in ['orbit', 'dolly_in']]),
]
out = []; n = 0
for title, intro, items in sections:
    cards = []
    for desc, ctrl, gen, ref, tag in items:
        if not Path(gen).exists(): continue
        sbs = RT / f'cloud/sbs/{tag}_sbs.mp4'
        if sbs.exists():
            sn = f'{tag}_sbs.mp4'; put(sbs, sn); cols = [f'<figure class="wide"><video src="assets/samples/{sn}" muted loop autoplay playsinline preload="metadata"></video><figcaption>input control video (left) and generated video (right), one file, frame-aligned</figcaption></figure>']
        else:
            cn = f'control_{tag}.mp4'; gn = f'{tag}.mp4'; put(ctrl, cn); put(gen, gn); cols = [vid(cn, 'control video'), vid(gn, 'generated')]
        if ref is not None and Path(ref).exists():
            rn = f'ref_{tag}' + Path(ref).suffix; put(ref, rn); cols.append(vid(rn, 'original Veo clip') if rn.endswith('.mp4') else img(rn, 'reference image'))
        cards.append(f'<div class="case"><p class="desc">{html.escape(desc)}</p><div class="row">{"".join(cols)}</div></div>'); n += 1
    if cards: out.append(f'<h2>{html.escape(title)}</h2><p class="note">{html.escape(intro)}</p>{"".join(cards)}')
for f in ['F1_teaser.png', 'F4_strip.png', 'F6_camera_paths.png']:
    src = Path('/Users/george_yang/workspace/doc/wan_control_demo/paper/figures') / f
    if src.exists(): shutil.copy2(src, A / 'figures' / f)
page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoaG: Cylinders on a Grid</title>
<style>
:root{{--fg:#1b1b1b;--muted:#666;--bg:#fff;--line:#e6e6e6}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);line-height:1.5}}
main{{max-width:1180px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:34px;margin:0 0 4px}} .sub{{font-size:19px;color:var(--muted);margin:0 0 10px}}
.links a{{display:inline-block;margin:6px 10px 0 0;padding:6px 12px;border:1px solid var(--line);border-radius:8px;text-decoration:none;color:var(--fg)}} .links a:hover{{background:#f5f5f5}}
h2{{font-size:22px;margin:40px 0 6px;border-bottom:1px solid var(--line);padding-bottom:6px}}
p.lead{{font-size:17px}} img.full{{width:100%;height:auto;display:block;border:1px solid var(--line)}}
.case{{margin:14px 0 22px}} .desc{{margin:0 0 6px;color:var(--fg);font-size:14px}}
.row{{display:flex;gap:10px;flex-wrap:wrap}} figure{{margin:0;flex:1 1 340px;max-width:380px}} figure.wide{{flex:2 1 700px;max-width:780px}} figure.wide video{{aspect-ratio:32/9}}
video,figure img{{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;display:block;border-radius:4px}}
figcaption{{font-size:12px;color:var(--muted);margin-top:3px}} .note{{font-size:14px;color:var(--muted);margin:0 0 8px}}
footer{{margin-top:60px;font-size:13px;color:var(--muted)}}
</style></head>
<body><main>
<h1>CoaG: Cylinders on a Grid</h1>
<p class="sub">Coarse 3D layout control for video generation</p>
<p>Zhangsihao Yang<span class="note"> &middot; research preview, September 2026</span></p>
<p class="links"><a href="CoaG_draft.pdf">Paper (draft PDF)</a><a href="https://github.com/zshyang/CoaG">Code + editor</a><a href="https://huggingface.co/zshyang1106/CoaG-Wan2.2-Fun-A14B-Control-LoRA">LoRA weights</a><a href="#data">Data</a></p>
<p class="lead">A user draws the crudest possible 3D scene, a ground grid and one cylinder per person, and moves the cylinders and the camera over 81 frames. A LoRA on Wan2.2-Fun-Control turns that sketch into a photoreal video in which the people stand where the cylinders stand, move as the cylinders move, and the camera moves as the drawn camera moves. Appearance comes from the text and a background reference image; layout and motion come from the geometry. The training pairs come from an automatic engine that lifts text-to-video output back to its geometry, with no real footage and no manual labels.</p>
<img class="full" src="assets/figures/F1_teaser.png" alt="teaser">
<p class="note">Left: the control video (three of 81 frames). Right: generated videos. The rows share the geometry and differ only in the text (row 2) or the background reference image (row 3).</p>
<h2>How the training pairs are made</h2>
<img class="full" src="assets/figures/F4_strip.png" alt="data engine">
<p class="note">One clip through the data engine: input frame, person masks (SAM 3.1), background (LaMa), ground mask (agent loop over SAM 3 phrases), plane and cylinders (HunyuanWorld-Mirror cameras and points, RANSAC plane), control frame.</p>
{"".join(out)}
<p class="note">All videos: 480 x 832, 81 frames at 16 fps, 50 sampling steps, LoRA weight 0.55 on both experts, one fixed seed, no cherry-picking within a case. Videos loop; hover to see the controls.</p>
<h2 id="data">Data and weights</h2>
<p class="note">LoRA weights (both experts): Hugging Face, link above. Captions and seed cards: in the code repository. Raw Veo clips, 81-frame clips, control videos and background images: Google Drive, link to be added. Colosseum background: "Colosseum Interior 1" by daryl_mitchell, CC BY-SA 2.0, via Wikimedia Commons (cropped, tourists patched out).</p>
<footer>Built on Wan2.2-Fun-A14B-Control and VideoX-Fun (Alibaba PAI), SAM 3 (Meta), LaMa, HunyuanWorld-Mirror (Tencent), GeoCalib, Veo 3.1 (Google DeepMind via fal.ai). Compute: RunPod.</footer>
</main></body></html>'''
(HERE / 'index.html').write_text(page); print('index.html written with', n, 'cases;', sum(f.stat().st_size for f in (A / 'samples').iterdir()) // 2**20, 'MB of assets')
