"""Selective, resumable download of Mengyi's full_run Drive folder (public link, no auth) via gdown.

Per clip we fetch only what step 4 and training need (tracks.json, ground_points.npz, cams10.npz, background.png,
masks, qc/agent json); clip81.mp4 (we have it), frame0.png and the overlay videos are skipped (saves ~5 MB/clip).
Usage: python fetch_mengyi_full.py <out_dir> [--ids c0000,c0001 | --range 0-1999] [--threads 8]
Verifies each file after download (json parses, npz loads, png opens); reruns skip verified files.
"""
import sys, os, json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, gdown, numpy as np
from PIL import Image
from gdown.download_folder import _parse_embedded_folder_view as P

ROOT = '1TEpTZiue-poikgt5mkXiOvL3RyGIIs8s'
WANT = {'tracks.json', 'ground_points.npz', 'cams10.npz', 'background.png', 'background_mask.png', 'ground_mask.png',
        'ground_agent.json', 'clip81.txt', 'qc_step0.json', 'qc_step1.json', 'qc_step2.json', 'qc_step3.json', 'qc_step3a.json'}
local = threading.local()

def sess():
    if not hasattr(local, 's'):
        local.s = requests.Session(); local.s.headers['User-Agent'] = 'Mozilla/5.0'
    return local.s

def listing(fid, tries=4):
    for t in range(tries):
        try:
            return P(sess=sess(), folder_id=fid, verify=True)[1]
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))

def ok(path):
    try:
        if path.endswith('.json'):
            json.load(open(path))
        elif path.endswith('.npz'):
            np.load(path).files
        elif path.endswith('.png'):
            Image.open(path).verify()
        return os.path.getsize(path) > 0
    except Exception:
        return False

def fetch_clip(cid, fid, out):
    d = f'{out}/{cid}'; os.makedirs(d, exist_ok=True)
    need = [n for n in WANT if not (os.path.exists(f'{d}/{n}') and ok(f'{d}/{n}'))]
    if not need:
        return cid, 0, []
    files = {c[1]: c[0] for c in listing(fid)}
    missing = [n for n in need if n not in files]
    got = 0
    for n in need:
        if n not in files:
            continue
        for t in range(3):
            try:
                gdown.download(id=files[n], output=f'{d}/{n}', quiet=True, use_cookies=False)
                if ok(f'{d}/{n}'):
                    got += 1; break
            except Exception:
                time.sleep(3 * (t + 1))
        else:
            missing.append(n)
    return cid, got, missing

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = args[0]
    opt = {a.split('=')[0]: a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--') and '=' in a}
    threads = int(opt.get('--threads', 8))
    top = listing(ROOT)
    clips = {c[1]: c[0] for c in top if c[1].startswith('c') and len(c[1]) == 5}
    others = {c[1]: c[0] for c in top if c[1] in ('summary.md', '_reports')}
    if '--ids' in opt:
        want_ids = opt['--ids'].split(',')
    elif '--range' in opt:
        a, b = map(int, opt['--range'].split('-')); want_ids = [f'c{i:04d}' for i in range(a, b + 1)]
    else:
        want_ids = sorted(clips)
    todo = [(i, clips[i]) for i in want_ids if i in clips]
    absent = [i for i in want_ids if i not in clips]
    print(f'drive has {len(clips)} clip folders; fetching {len(todo)}; not yet uploaded: {len(absent)}', flush=True)
    t0 = time.time(); done = 0; bad = []
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(fetch_clip, cid, fid, out): cid for cid, fid in todo}
        for f in as_completed(futs):
            try:
                cid, got, missing = f.result()
                if missing:
                    bad.append((cid, missing))
            except Exception as e:
                bad.append((futs[f], [f'ERR {e}']))
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f'{done}/{len(todo)} clips  {time.time() - t0:.0f}s  problems {len(bad)}', flush=True)
    if 'summary.md' in others:
        gdown.download(id=others['summary.md'], output=f'{out}/summary.md', quiet=True, use_cookies=False)
    json.dump(dict(absent=absent, bad=bad), open(f'{out}/fetch_status.json', 'w'), indent=1)
    print('problems:', bad[:10], '| not uploaded:', len(absent), flush=True)

if __name__ == '__main__':
    main()
