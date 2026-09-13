"""Measure real camera motion per clip from the WorldMirror keyframe cameras and write it next to the training metadata.

static = camera path < 0.25 subject heights AND max rotation < 3 deg across the 10 keyframes.
Writes <train_data>/camera_motion.json {id: {path, rot_deg, static, label}} and metadata_balanced_camera.json
(every measured-moving clip duplicated once, so moving : static is about 2 : 1 instead of 1 : 1).
Usage: python camera_motion_stats.py <train_data_dir> <mengyi_full_dir> <step4_out_dir> <captions.jsonl>
"""
import sys, os, json, collections, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocess')); from geom import cam_center
D, M, O, CAP = sys.argv[1:5]
caps = {}
for l in open(CAP):
    r = json.loads(l); caps[r['id']] = r.get('seed', r).get('camera')
meta = json.load(open(f'{D}/metadata.json'))
stats, out, lab, mov = {}, [], collections.Counter(), collections.Counter()
for m in meta:
    cid = m['id']; c = np.load(f'{M}/{cid}/cams10.npz'); w2c = c['w2c']; p = json.load(open(f'{O}/{cid}/plane.json'))
    C = np.stack([cam_center(w) for w in w2c]); h = p['grid_spacing'] or 1.0
    path = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum() / h)
    R0 = w2c[0][:, :3]; rot = max(float(np.degrees(np.arccos(np.clip((np.trace(R0 @ w[:, :3].T) - 1) / 2, -1, 1)))) for w in w2c)
    static = bool(path < 0.25 and rot < 3.0)
    stats[cid] = dict(path_subject_heights=round(path, 3), rot_deg=round(rot, 2), static=static, label=caps[cid])
    lab[caps[cid]] += 1; mov[caps[cid]] += (not static)
    row = dict(m, camera_static=static); out.append(row)
    if not static: out.append(row)          # duplicate moving clips
json.dump(stats, open(f'{D}/camera_motion.json', 'w'), indent=1)
json.dump(out, open(f'{D}/metadata_balanced_camera.json', 'w'), indent=1)
n = len(meta); s = sum(v['static'] for v in stats.values())
print(f'{n} clips: static {s} ({100*s/n:.0f}%), moving {n-s}; balanced metadata rows {len(out)} (moving share {100*2*(n-s)/len(out):.0f}%)')
for label, cnt in lab.most_common():
    print(f'  {cnt:4d}  {100*mov[label]/cnt:3.0f}% moving  {label}')
