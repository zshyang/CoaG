"""Completeness check of the ingested full_run data. Usage: python check_mengyi_full.py <mengyi_full_dir> [n_expected=2000]
Prints counts and writes <dir>/_missing.json = {cid: [missing files]} for clips with any of the 13 needed files absent or unreadable."""
import sys, os, json
import numpy as np
from PIL import Image
WANT = ['tracks.json', 'ground_points.npz', 'cams10.npz', 'background.png', 'background_mask.png', 'ground_mask.png',
        'ground_agent.json', 'clip81.txt', 'qc_step0.json', 'qc_step1.json', 'qc_step2.json', 'qc_step3.json', 'qc_step3a.json']
def ok(p):
    try:
        if p.endswith('.json'): json.load(open(p))
        elif p.endswith('.npz'): np.load(p).files
        elif p.endswith('.png'): Image.open(p).verify()
        return os.path.getsize(p) > 0
    except Exception:
        return False
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
missing, absent = {}, []
for i in range(n):
    cid = f'c{i:04d}'
    if not os.path.isdir(f'{d}/{cid}'):
        absent.append(cid); continue
    m = [w for w in WANT if not (os.path.exists(f'{d}/{cid}/{w}') and ok(f'{d}/{cid}/{w}'))]
    if m: missing[cid] = m
json.dump(dict(absent=absent, missing=missing), open(f'{d}/_missing.json', 'w'), indent=1)
print(f'complete {n - len(absent) - len(missing)} / {n}; absent {len(absent)}; incomplete {len(missing)}')
if missing: print('incomplete sample:', list(missing.items())[:5])
if absent: print('absent range:', absent[0], '..', absent[-1])
