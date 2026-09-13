"""Extract the needed per-clip files from Google Drive folder-download zips into <out_dir>/<cid>/.

Drive zips a folder download as c0010/<file> (single folder) or full_run/c0010/<file> (whole folder, split into -001.zip,
-002.zip, ...). We keep only what step 4 and training need and verify each file (json parses, npz loads, png opens).
Usage: python ingest_drive_zips.py <out_dir> <zip_or_dir> [more zips...] [--delete]   (--delete removes a zip after a clean extract)
"""
import sys, os, json, zipfile, glob, io
import numpy as np
from PIL import Image

WANT = {'tracks.json', 'ground_points.npz', 'cams10.npz', 'background.png', 'background_mask.png', 'ground_mask.png',
        'ground_agent.json', 'clip81.txt', 'qc_step0.json', 'qc_step1.json', 'qc_step2.json', 'qc_step3.json', 'qc_step3a.json'}
TOP = {'summary.md'}

def ok(path):
    try:
        if path.endswith('.json'): json.load(open(path))
        elif path.endswith('.npz'): np.load(path).files
        elif path.endswith('.png'): Image.open(path).verify()
        return os.path.getsize(path) > 0
    except Exception:
        return False

def ingest(zp, out):
    n_ok, bad, clips = 0, [], set()
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split('/')
            name = parts[-1]
            cid = next((p for p in parts[:-1] if len(p) == 5 and p.startswith('c') and p[1:].isdigit()), None)
            if '_reports' in parts and name.endswith('.jsonl'):
                dst = f'{out}/_reports/{name}'
            elif cid and name in WANT:
                dst = f'{out}/{cid}/{name}'
            elif name in TOP and cid is None:
                dst = f'{out}/{name}'
            else:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(info) as src, open(dst, 'wb') as f:
                f.write(src.read())
            if dst.endswith(('.json', '.npz', '.png')) and not ok(dst):
                bad.append(info.filename); os.remove(dst)
            else:
                n_ok += 1
            if cid: clips.add(cid)
    return n_ok, bad, clips

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out, srcs = args[0], args[1:]
    zips = []
    for s in srcs:
        zips += sorted(glob.glob(f'{s}/*.zip')) if os.path.isdir(s) else [s]
    total_clips = set()
    for zp in zips:
        n_ok, bad, clips = ingest(zp, out)
        total_clips |= clips
        print(f'{os.path.basename(zp)}: {n_ok} files, {len(clips)} clips, bad {len(bad)}', flush=True)
        if bad:
            print('  bad:', bad[:5])
        elif '--delete' in sys.argv:
            os.remove(zp)
    print('clips touched:', len(total_clips))

if __name__ == '__main__':
    main()
