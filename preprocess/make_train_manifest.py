"""Build train_manifest.jsonl from step-4 outputs.
Usage: python make_train_manifest.py <mengyi_full_dir> <step4_out_dir> <out_jsonl> [--soft=count_mismatch]
--soft=FLAG,FLAG demotes the named hard flags to soft when deciding training_ready (policy decision, no re-render).
Adds soft flag `bystanders` when step 4 found >= expected + 5 subjects (crowd / audience got cylinders).
Row: id, caption, video (clip81), control (control.mp4), ref_image (LaMa background.png), training_ready, hard_flags,
soft_flags, holdout (ids ending in 49 or 99 are never trained on)."""
import sys, os, json
WS = '/Users/george_yang/workspace'
args = [a for a in sys.argv[1:] if not a.startswith('--')]
mdir, odir, out = args[:3]
soft = set(next((a.split('=',1)[1] for a in sys.argv[1:] if a.startswith('--soft=')), '').split(',')) - {''}
caps = {}
for l in open(f'{WS}/code/wan_control_demo/captions/out/captions.jsonl'):
    r = json.loads(l); caps[r['id']] = r['caption']
rows, n_ready, n_hold = [], 0, 0
for cid in sorted(caps):
    q = f'{odir}/{cid}/qc.json'
    if not os.path.exists(q):
        continue
    qc = json.load(open(q))
    bg = f'{mdir}/{cid}/background.png'
    hard = [f for f in qc.get('hard_flags', []) if f not in soft]
    softf = list(qc.get('soft_flags', [])) + [f for f in qc.get('hard_flags', []) if f in soft]
    if qc.get('expected_persons') is not None and qc.get('main_subjects_final', 0) >= qc['expected_persons'] + 5:
        softf.append('bystanders')
    ready = (not hard) and os.path.exists(bg) and os.path.exists(f'{odir}/{cid}/control.mp4')
    hold = cid[-2:] in ('49', '99')
    rows.append(dict(id=cid, caption=caps[cid], video=f'{WS}/runtime/wan_control_demo/clip81/{cid}.mp4',
                     control=f'{odir}/{cid}/control.mp4', ref_image=bg, training_ready=ready,
                     hard_flags=hard, soft_flags=sorted(set(softf)), holdout=hold))
    n_ready += ready and not hold; n_hold += ready and hold
with open(out, 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
print(f'rows {len(rows)}  train-ready {n_ready}  holdout-ready {n_hold}  -> {out}')
