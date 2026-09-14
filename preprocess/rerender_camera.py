"""Re-render a clip's control video (same plane, same cylinder trajectories) under synthetic camera paths.

Paths, all 81 frames, expressed relative to the clip's own camera 0 and plane frame:
  static     camera 0 held fixed (control for the experiment)
  dolly_in   move forward along the ground-projected view direction by min(1.2 subject heights, 40% of the distance to the subjects), no re-aim
  dolly_out  the reverse
  orbit      swing -20..+20 deg around the plane normal through the subjects' centroid, always aimed at it
  pan        camera fixed, yaw -12..+12 deg about the plane normal
  crane      start 1.5 subject heights higher, aimed at the centroid, descend to the original height
Usage: python rerender_camera.py <preproc_dir> <mengyi_dir> <out_dir> <cid> [names...]
Writes <out_dir>/<cid>/control_<name>.mp4 (1280x720, 16 fps, 81 frames) and <out_dir>/<cid>/cams_<name>.json.
"""
import sys, os, json, subprocess, tempfile, shutil
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import cam_center, project

W, H, SS, FPS, NF = 1280, 720, 2, 16, 81
NAMES = ['static', 'dolly_in', 'dolly_out', 'orbit', 'pan', 'crane']

def hex2rgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
def unit(v): return v / np.linalg.norm(v)
def rot_axis(axis, ang):
    a = unit(axis); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
def look_at(C, T, n):
    z = unit(T - C); x = unit(np.cross(-n, z)); y = np.cross(z, x)
    R = np.stack([x, y, z]); return np.hstack([R, (-R @ C)[:, None]])
def w2c_from(R, C): return np.hstack([R, (-R @ C)[:, None]])

def convex_hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2: return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]

def encode(frames, path):
    tmp = tempfile.mkdtemp()
    for i, im in enumerate(frames): im.save(f'{tmp}/{i:03d}.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', f'{tmp}/%03d.png', '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', path], check=True)
    shutil.rmtree(tmp)

def camera_paths(name, C0, R0, n, o, u, v, T, h):
    f0 = R0.T @ np.array([0, 0, 1.0]); fh = unit(f0 - (f0 @ n) * n)
    dist = float((T - C0) @ fh)                       # ground distance from the camera to the subjects' centroid
    D = min(1.2 * h, 0.4 * dist)                        # dolly travel: never more than 40% of the way to the subjects
    ts = np.linspace(0, 1, NF); out = []
    for t in ts:
        if name == 'static': out.append(w2c_from(R0, C0))
        elif name == 'dolly_in': out.append(w2c_from(R0, C0 + t * D * fh))
        elif name == 'dolly_out': out.append(w2c_from(R0, C0 - t * D * fh))
        elif name == 'orbit':
            Rm = rot_axis(n, np.radians(40 * (t - 0.5))); C = T + Rm @ (C0 - T); out.append(look_at(C, T, n))
        elif name == 'pan':
            Rm = rot_axis(n, np.radians(24 * (t - 0.5))); out.append(w2c_from(R0 @ Rm.T, C0))
        elif name == 'crane':
            C = C0 + (1 - t) * 1.5 * h * n; out.append(look_at(C, T, n))
    return out

def render(K, w2cs, n, o, u, v, cyl, s_grid, pos_uv):
    (u0, v0), (u1, v1) = pos_uv.min(0) - 6 * s_grid, pos_uv.max(0) + 6 * s_grid
    us = np.arange(np.floor(u0 / s_grid) * s_grid, u1 + 1e-9, s_grid); vs = np.arange(np.floor(v0 / s_grid) * s_grid, v1 + 1e-9, s_grid)
    frames = []; samp = np.linspace(0, 1, 60); ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for k in range(NF):
        im = Image.new('RGB', (W * SS, H * SS), (0, 0, 0)); dr = ImageDraw.Draw(im); Kk = K.copy(); Kk[:2] *= SS; w2c = w2cs[k]
        zmin = 0.05 * s_grid
        def polyline(P3):
            uv, z = project(Kk, w2c, P3); ok = z > zmin; run = []
            for i in range(len(P3)):
                if ok[i]: run.append(tuple(uv[i]))
                else:
                    if len(run) > 1: dr.line(run, fill=(255, 255, 255), width=2 * SS)
                    run = []
            if len(run) > 1: dr.line(run, fill=(255, 255, 255), width=2 * SS)
        for a in us: polyline(np.stack([o + a * u + (v0 + (v1 - v0) * t) * v for t in samp]))
        for b in vs: polyline(np.stack([o + (u0 + (u1 - u0) * t) * u + b * v for t in samp]))
        Ck = cam_center(w2c)
        order = sorted(range(len(cyl)), key=lambda i: -np.linalg.norm(np.array(cyl[i]['pos'][k]) - Ck))
        for i in order:
            c = cyl[i]; p = np.array(c['pos'][k]); r = c['radius']; hh = c['height']
            ring = np.stack([p + r * (np.cos(a) * u + np.sin(a) * v) for a in ang]); pts3 = np.vstack([ring, ring + hh * n])
            uv, z = project(Kk, w2c, pts3)
            if (z <= zmin).any(): continue
            hull = convex_hull([tuple(q) for q in uv]); col = hex2rgb(c['color']); dark = tuple(int(x * 0.6) for x in col)
            if len(hull) >= 3: dr.polygon(hull, fill=col, outline=dark, width=SS)
        frames.append(im.resize((W, H), Image.BOX))
    return frames

def main():
    pdir, mdir, out, cid = sys.argv[1:5]; names = sys.argv[5:] or NAMES
    p = json.load(open(f'{pdir}/{cid}/plane.json')); cyl = json.load(open(f'{pdir}/{cid}/cylinders.json'))['subjects']
    cams = np.load(f'{mdir}/{cid}/cams10.npz'); K = cams['K'][0].astype(float); w2c0 = cams['w2c'][0].astype(float)
    if os.environ.get('ZOOM'):  # optional uniform zoom about the principal point (used to match a reference image cropped free of letterbox bars)
        K[0, 0] *= float(os.environ['ZOOM']); K[1, 1] *= float(os.environ['ZOOM'])
    n, o, u, v = (np.array(p[k]) for k in ('normal', 'origin', 'u', 'v')); h = p['grid_spacing']
    C0 = cam_center(w2c0); R0 = w2c0[:, :3]
    allpos = np.array([c['pos'] for c in cyl])                      # [S, 81, 3]
    T = allpos[:, 40].mean(0) + 0.5 * h * n
    pos_uv = np.stack([((allpos.reshape(-1, 3) - o) @ u), ((allpos.reshape(-1, 3) - o) @ v)], 1)
    pos_uv = np.vstack([pos_uv, [[(C0 - o) @ u, (C0 - o) @ v]]])
    os.makedirs(f'{out}/{cid}', exist_ok=True)
    for name in names:
        w2cs = camera_paths(name, C0, R0, n, o, u, v, T, h)
        frames = render(K, w2cs, n, o, u, v, cyl, h, pos_uv)
        encode(frames, f'{out}/{cid}/control_{name}.mp4')
        json.dump(dict(K=K.tolist(), w2c=[w.tolist() for w in w2cs], path=name), open(f'{out}/{cid}/cams_{name}.json', 'w'))
        print(cid, name, 'done', flush=True)

if __name__ == '__main__':
    main()
