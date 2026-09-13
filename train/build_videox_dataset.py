"""Stage the training set in VideoX-Fun layout from train_manifest.jsonl.

Output dir:
  videos/<id>.mp4      target clip (clip81: 81 frames, 16 fps, 1280x720)   [hard link to runtime/.../clip81]
  control/<id>.mp4     control video (plane grid + cylinders), same length  [hard link]
  ref/<id>.png         LaMa background = reference image                    [hard link]
  metadata.json        VideoX-Fun fields (file_path, control_file_path, text, type, width, height) + ref_file_path (our patch)
  metadata_no_bystanders.json   same but without clips flagged `bystanders`
  holdout.json         the 40 held-out ids (never trained on)
Usage: python build_videox_dataset.py <train_manifest.jsonl> <out_dir>
"""
import sys, os, json
man, out = sys.argv[1], sys.argv[2]
for d in ('videos', 'control', 'ref'):
    os.makedirs(f'{out}/{d}', exist_ok=True)

def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil; shutil.copy2(src, dst)

rows = [json.loads(l) for l in open(man)]
meta, meta_nb, hold = [], [], []
for r in rows:
    if not r['training_ready']:
        continue
    cid = r['id']
    link(r['video'], f'{out}/videos/{cid}.mp4'); link(r['control'], f'{out}/control/{cid}.mp4'); link(r['ref_image'], f'{out}/ref/{cid}.png')
    if r['holdout']:
        hold.append(cid); continue
    m = dict(file_path=f'videos/{cid}.mp4', control_file_path=f'control/{cid}.mp4', ref_file_path=f'ref/{cid}.png',
             text=r['caption'], type='video', width=1280, height=720, id=cid, soft_flags=r['soft_flags'])
    meta.append(m)
    if 'bystanders' not in r['soft_flags']:
        meta_nb.append(m)
json.dump(meta, open(f'{out}/metadata.json', 'w'), indent=1)
json.dump(meta_nb, open(f'{out}/metadata_no_bystanders.json', 'w'), indent=1)
json.dump(hold, open(f'{out}/holdout.json', 'w'))
print(f'train rows {len(meta)} (no-bystanders {len(meta_nb)}), holdout {len(hold)} -> {out}')
