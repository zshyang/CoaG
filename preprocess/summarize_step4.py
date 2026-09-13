"""Consolidate step-4 outputs: one row per clip from <out_dir>/<cid>/qc.json -> <out_dir>/step4_report.jsonl + flag statistics.
Usage: python summarize_step4.py <step4_out_dir> [n_expected=2000]"""
import sys, os, json, collections
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rows, missing = [], []
for i in range(n):
    cid = f'c{i:04d}'; p = f'{d}/{cid}/qc.json'
    if os.path.exists(p): rows.append(json.load(open(p)))
    else: missing.append(cid)
with open(f'{d}/step4_report.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
hard = collections.Counter(x for r in rows for x in r['hard_flags']); soft = collections.Counter(x for r in rows for x in r['soft_flags'])
ready = sum(r['training_ready'] for r in rows)
ready_soft_cm = sum(1 for r in rows if not (set(r['hard_flags']) - {'count_mismatch'}))
print(f'clips with qc: {len(rows)} / {n}; missing qc: {len(missing)}' + (f' (e.g. {missing[:5]})' if missing else ''))
print(f'training_ready (current rule): {ready}  |  if count_mismatch were soft: {ready_soft_cm}')
print('hard:', dict(hard)); print('soft:', dict(soft))
cm = [r for r in rows if 'count_mismatch' in r['hard_flags']]
print('count_mismatch found-expected:', sorted(collections.Counter(r['main_subjects_final'] - r['expected_persons'] for r in cm).items()))
print('plane inlier ratio < 0.5:', sum(1 for r in rows if r['inlier_ratio'] < 0.5), '| cam_height_ratio<0.4:', sum(1 for r in rows if (r.get('cam_height_ratio') or 1) < 0.4), '| horizon_diff>60:', sum(1 for r in rows if (r.get('horizon_diff_px') or 0) > 60))
json.dump(dict(missing=missing), open(f'{d}/_missing_qc.json', 'w'))
