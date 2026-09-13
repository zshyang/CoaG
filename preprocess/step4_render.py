"""Step 4 (George, local): plane fit -> camera interpolation -> cylinders -> control video -> preview -> QC.

Usage: python step4_render.py <mengyi_dir> <clip81_dir> <out_dir> [ids...] [--skip-done] [--no-preview] [--tag=NAME] [--geocalib=geocalib.json]
  mengyi_dir/<id>/{tracks.json, ground_points.npz, cams10.npz, qc_step*.json}
  clip81_dir/<id>.mp4            George's 81-frame 16 fps training targets
  out_dir/<id>/{plane.json, cylinders.json, control.mp4, preview.mp4, qc.json}
Rendering spec = PREPROCESS_DESIGN.md Decision V1 / 04_cylinders_control_video_qc.md.
"""
import sys, os, json, subprocess, shutil, tempfile
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import fit_ground, interpolate_cams, project, pixel_ray, ray_plane, cam_center, horizon_y

W, H, SS = 1280, 720, 2                 # canvas and supersampling
FPS, NF = 16, 81
ANCHOR = range(16)
PALETTE = ['#E53935', '#1E88E5', '#43A047', '#FDD835', '#8E24AA', '#FB8C00']
HARD = ('ground_fail', 'plane_fail', 'count_mismatch', 'no_anchor')


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def convex_hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def load_clip_frames(path, n=NF):
    tmp = tempfile.mkdtemp()
    subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vsync', '0', f'{tmp}/%03d.png'], check=True)
    frames = [Image.open(f'{tmp}/{i + 1:03d}.png').convert('RGB') for i in range(n)]
    shutil.rmtree(tmp)
    return frames


def encode(frames, path, fps=FPS, crf=14):
    tmp = tempfile.mkdtemp()
    for i, im in enumerate(frames):
        im.save(f'{tmp}/{i:03d}.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(fps), '-i', f'{tmp}/%03d.png', '-c:v', 'libx264',
                    '-crf', str(crf), '-pix_fmt', 'yuv420p', path], check=True)
    shutil.rmtree(tmp)


def process(cid, mdir, clip81_dir, odir, preview=True, geocalib=None):
    d = f'{mdir}/{cid}'
    os.makedirs(f'{odir}/{cid}', exist_ok=True)
    tr = json.load(open(f'{d}/tracks.json'))
    gp = np.load(f'{d}/ground_points.npz'); cams = np.load(f'{d}/cams10.npz')
    qc_prev = {}
    for k in (1, 2, 3):
        p = f'{d}/qc_step{k}.json'
        if os.path.exists(p):
            qc_prev[f'step{k}'] = json.load(open(p))
    expected = tr.get('expected_persons', qc_prev.get('step1', {}).get('expected_persons'))
    md = float(gp['median_depth'])
    K10, w2c10, kf = cams['K'].astype(float), cams['w2c'].astype(float), cams['keyframes']

    # ---- plane
    normals = gp['normals'].astype(float) if 'normals' in gp.files else None
    g = fit_ground(gp['points'].astype(float), md, w2c10[0], normals=normals)
    n, dpl, o, u, v = g['normal'], g['d'], g['origin'], g['u'], g['v']
    flags = set()
    if qc_prev.get('step3', {}).get('ground_fallback'):
        flags.add('ground_fail')
    if not g['plane_ok']:
        flags.add('plane_fail')
    K, w2c = interpolate_cams(K10, w2c10, kf, NF)

    # ---- subjects
    # final main-subject rule: step-1 prelim flag (tall in the anchor second) OR a track that is visible for at least half
    # the clip and at least 40% as tall as the tallest track (people who enter after the first second)
    def med_h(t):
        hs = sorted(f['height'] for f in t['frames'] if f['visible'] and f.get('height'))
        return hs[len(hs) // 2] if hs else 0
    live = [t for t in tr['tracks'] if not t.get('dropped_reflection')]
    h_max = max([med_h(t) for t in live], default=0)
    cands = [t for t in live if t.get('main_subject_prelim') or
             (sum(f['visible'] for f in t['frames']) >= NF // 2 and med_h(t) >= max(72, 0.4 * h_max))]
    subjects = []

    def foot_hits(fr, ks):
        out = []
        for k in ks:
            f = fr[k]
            if f['visible'] and not f['touches_bottom'] and f.get('foot'):
                C, dw = pixel_ray(K[k], w2c[k], f['foot'][0], f['foot'][1])
                P = ray_plane(C, dw, n, dpl)
                if P is not None:
                    out.append((k, P, f))
        return out

    for t in cands:
        fr = t['frames']
        # height from frames with visible feet: prefer the anchor second, else fall back to the whole clip
        hits = foot_hits(fr, ANCHOR)
        source = 'anchor'
        if len(hits) < 4:
            hits = foot_hits(fr, range(NF))
            source = 'clip'
        if len(hits) < 3:
            continue
        hs = []
        for k, P, f in hits:
            z = (w2c[k][:, :3] @ P + w2c[k][:, 3])[2]
            hs.append(f['height'] * z / K[k][1, 1])
        height = float(np.median(hs))
        subjects.append(dict(track=t, height=height, radius=0.2 * height, anchor_source=source))
    if not subjects:
        flags.add('no_anchor')
    def first_foot_x(s):                    # left-to-right order by the first visible foot (frame 0 may be invisible)
        for f in s['track']['frames']:
            if f.get('visible') and f.get('foot'):
                return f['foot'][0]
        return 0.0
    subjects.sort(key=first_foot_x)
    if expected is not None and len(subjects) != expected:
        flags.add('count_mismatch')

    # ---- trajectories
    cyl = []
    for si, s in enumerate(subjects):
        fr = s['track']['frames']; h = s['height']
        pos = np.full((NF, 3), np.nan); method = [''] * NF; feet = [False] * NF
        for k in range(NF):
            f = fr[k]
            if not f['visible']:
                continue
            if not f['touches_bottom']:
                C, dw = pixel_ray(K[k], w2c[k], f['foot'][0], f['foot'][1])
                P = ray_plane(C, dw, n, dpl)
                if P is not None:
                    pos[k] = P; method[k] = 'foot_ray'; feet[k] = True
            else:                                  # head ray: point at height h above the plane, dropped to the plane
                x0, y0, x1, y1 = f['box']
                C, dw = pixel_ray(K[k], w2c[k], 0.5 * (x0 + x1), y0)
                denom = n @ dw
                if abs(denom) > 1e-9:
                    sdist = (h - (n @ C + dpl)) / denom
                    if sdist > 0:
                        pos[k] = C + sdist * dw - h * n; method[k] = 'head_ray'
        vis = ~np.isnan(pos[:, 0])
        idx = np.arange(NF)
        for j in range(3):
            pos[:, j] = np.interp(idx, idx[vis], pos[vis, j])
        interp = (~vis).tolist()
        sm = gaussian_filter1d(pos, sigma=2, axis=0, mode='nearest')
        jumps = int((np.linalg.norm(np.diff(sm, axis=0), axis=1) > h).sum())
        cyl.append(dict(track_id=s['track']['track_id'], height=h, radius=s['radius'], color=PALETTE[si % len(PALETTE)],
                        anchor_source=s['anchor_source'],
                        pos=sm.tolist(), method=method, feet_visible=feet, interpolated=interp, track_jumps=jumps))
    soft = set()
    if any(c['track_jumps'] for c in cyl):
        soft.add('track_jump')
    if any(c['anchor_source'] == 'clip' for c in cyl):
        soft.add('anchor_late')            # some subject's height came from outside the first second
    feet_cut_later = float(np.mean([not fv for c in cyl for fv in c['feet_visible'][16:]])) if cyl else 0.0
    if feet_cut_later > 0.3:
        soft.add('feet_cut_later')

    # ---- render control video
    heights = [c['height'] for c in cyl]
    s_grid = float(np.median(heights)) if heights else g['camera_height'] / 3
    inl_uv = np.stack([(gp['points'][g['inliers']] - o) @ u, (gp['points'][g['inliers']] - o) @ v], 1)
    (u0, v0), (u1, v1) = inl_uv.min(0) - 2 * s_grid, inl_uv.max(0) + 2 * s_grid
    us = np.arange(np.floor(u0 / s_grid) * s_grid, u1 + 1e-9, s_grid)
    vs = np.arange(np.floor(v0 / s_grid) * s_grid, v1 + 1e-9, s_grid)
    zmin = 0.05 * md
    ctrl_frames = []
    for k in range(NF):
        im = Image.new('RGB', (W * SS, H * SS), (0, 0, 0)); dr = ImageDraw.Draw(im)
        Kk = K[k].copy(); Kk[:2] *= SS

        def draw_polyline(P3):
            uv, z = project(Kk, w2c[k], P3)
            ok = z > zmin
            run = []
            for i in range(len(P3)):
                if ok[i]:
                    run.append(tuple(uv[i]))
                else:
                    if len(run) > 1:
                        dr.line(run, fill=(255, 255, 255), width=2 * SS)
                    run = []
            if len(run) > 1:
                dr.line(run, fill=(255, 255, 255), width=2 * SS)
        samp = np.linspace(0, 1, 60)
        for a in us:
            draw_polyline(np.stack([o + a * u + (v0 + (v1 - v0) * t) * v for t in samp]))
        for b in vs:
            draw_polyline(np.stack([o + (u0 + (u1 - u0) * t) * u + b * v for t in samp]))
        # cylinders, far to near
        Ck = cam_center(w2c[k])
        order = sorted(range(len(cyl)), key=lambda i: -np.linalg.norm(np.array(cyl[i]['pos'][k]) - Ck))
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        for i in order:
            c = cyl[i]; p = np.array(c['pos'][k]); r = c['radius']; h = c['height']
            ring = np.stack([p + r * (np.cos(a) * u + np.sin(a) * v) for a in ang])
            pts3 = np.vstack([ring, ring + h * n])
            uv, z = project(Kk, w2c[k], pts3)
            if (z <= zmin).any():
                continue
            hull = convex_hull([tuple(q) for q in uv])
            col = hex2rgb(c['color']); dark = tuple(int(x * 0.6) for x in col)
            if len(hull) >= 3:
                dr.polygon(hull, fill=col, outline=dark, width=SS)
        ctrl_frames.append(im.resize((W, H), Image.BOX))
    encode(ctrl_frames, f'{odir}/{cid}/control.mp4')

    # ---- preview: clip81 | control | blend
    if preview:
        clip = load_clip_frames(f'{clip81_dir}/{cid}.mp4')
        prev = []
        for k in range(NF):
            a = clip[k].resize((640, 360), Image.BILINEAR); b = ctrl_frames[k].resize((640, 360), Image.BILINEAR)
            blend = Image.blend(a, b, 0.5)
            row = Image.new('RGB', (1920, 360)); row.paste(a, (0, 0)); row.paste(b, (640, 0)); row.paste(blend, (1280, 0))
            prev.append(row)
        encode(prev, f'{odir}/{cid}/preview.mp4', crf=20)

    # ---- plane sanity numbers (soft): horizon row in frame 0 and camera height in subject heights
    hy = horizon_y(K[0], w2c[0], n)
    cam_ratio = float(g['camera_height'] / np.median(heights)) if heights else None
    if cam_ratio is not None and cam_ratio < 0.4:
        soft.add('low_camera')          # camera below knee height: plane probably tilted toward the camera
    gc_y = (geocalib or {}).get(cid, {}).get('horizon_y_center')   # independent single-image horizon (GeoCalib, frame 0)
    horizon_diff = float(abs(hy - gc_y)) if (hy is not None and gc_y is not None) else None
    if horizon_diff is not None and horizon_diff > 60:
        soft.add('horizon_disagree')    # WorldMirror plane vs GeoCalib horizon differ by > 60 px (~3-4 deg)

    # ---- outputs
    json.dump(dict(normal=n.tolist(), d=dpl, inlier_ratio=g['inlier_ratio'], normal_angle_deg=g['normal_angle_deg'],
                   camera_height=g['camera_height'], origin=o.tolist(), u=u.tolist(), v=v.tolist(), median_depth=md,
                   grid_spacing=s_grid, normal_agreement=g['normal_agreement'], horizon_y_frame0=hy,
                   camera_height_over_subject_height=cam_ratio), open(f'{odir}/{cid}/plane.json', 'w'), indent=1)
    json.dump(dict(main_subjects=[c['track_id'] for c in cyl], subjects=cyl), open(f'{odir}/{cid}/cylinders.json', 'w'))
    qc = dict(id=cid, expected_persons=expected, main_subjects_final=len(cyl), count_match_final=(expected == len(cyl)),
              plane_ok=g['plane_ok'], inlier_ratio=round(g['inlier_ratio'], 3), normal_angle_deg=round(g['normal_angle_deg'], 1),
              normal_agreement=(round(g['normal_agreement'], 3) if g['normal_agreement'] is not None else None),
              horizon_y_frame0=(round(hy, 1) if hy is not None else None),
              cam_height_ratio=(round(cam_ratio, 2) if cam_ratio is not None else None),
              horizon_y_geocalib=(round(gc_y, 1) if gc_y is not None else None),
              horizon_diff_px=(round(horizon_diff, 1) if horizon_diff is not None else None),
              feet_cut_later_frac=round(feet_cut_later, 3), hard_flags=sorted(flags), soft_flags=sorted(soft),
              training_ready=not flags, prev=qc_prev)
    json.dump(qc, open(f'{odir}/{cid}/qc.json', 'w'), indent=1)
    return qc


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = [a for a in sys.argv[1:] if a.startswith('--')]
    skip_done = '--skip-done' in opts
    preview = '--no-preview' not in opts
    tag = next((a.split('=', 1)[1] for a in opts if a.startswith('--tag=')), '')
    gc_path = next((a.split('=', 1)[1] for a in opts if a.startswith('--geocalib=')), '')
    geocalib = json.load(open(gc_path)) if gc_path and os.path.exists(gc_path) else None
    mdir, clip81_dir, odir = args[:3]
    ids = args[3:] or sorted(d for d in os.listdir(mdir) if d.startswith('c') and len(d) == 5)
    os.makedirs(odir, exist_ok=True)
    report = f'{odir}/step4_report{"_" + tag if tag else ""}.jsonl'
    rows = []
    with open(report, 'a') as rep:
        for cid in ids:
            if skip_done and os.path.exists(f'{odir}/{cid}/qc.json'):
                continue
            try:
                q = process(cid, mdir, clip81_dir, odir, preview=preview, geocalib=geocalib)
                rows.append(q)
                rep.write(json.dumps(q) + '\n'); rep.flush()
                print(cid, 'subjects', q['main_subjects_final'], '/', q['expected_persons'], '| plane', q['plane_ok'],
                      '| hard', q['hard_flags'], '| soft', q['soft_flags'], '| READY' if q['training_ready'] else '| flagged', flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(cid, 'ERROR', e, flush=True)
                rep.write(json.dumps(dict(id=cid, error=str(e), training_ready=False)) + '\n'); rep.flush()
    print('training-ready: %d / %d' % (sum(q['training_ready'] for q in rows), len(rows)))
