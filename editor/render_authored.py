#!/usr/bin/env python3
"""Render an authored CoaG scene (editor/scene_schema.md) into an 81-frame control video.

Usage: .venv/bin/python editor/render_authored.py scene.json out_dir [--name NAME]
Writes out_dir/<name>/{control.mp4, cams.json, cylinders.json, plane.json, sheet.png}.

All drawing, camera-path and projection code is imported from the step-4 pipeline so authored videos match the training
distribution: rerender_camera.render / camera_paths / look_at / encode, geom.project, step4_render.PALETTE.
"""
import sys, os, json
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
import rerender_camera as rc                       # render(), camera_paths(), look_at(), encode(), NAMES
from geom import project, cam_center
from step4_render import PALETTE                   # colour order used for training (left-to-right)

NF = rc.NF


def unit(x):
    x = np.asarray(x, float); return x / np.linalg.norm(x)


def expand_scene(scene):
    """scene dict -> (K, w2cs[81], n, o, u, v, cyl list, s_grid, pos_uv)."""
    assert scene.get('n_frames', NF) == NF, 'the training data has 81 frames'
    W, H = scene.get('width', rc.W), scene.get('height', rc.H)
    assert (W, H) == (rc.W, rc.H), 'rerender_camera renders 1280x720 like the training control videos'
    pl = scene.get('plane', {})
    o = np.array(pl.get('origin', [0, 0, 0]), float); u = unit(pl.get('u', [1, 0, 0])); n = unit(pl.get('normal', [0, 0, 1]))
    v = np.cross(n, u)                                                     # plane_frame convention: v = n x u
    cam = scene['camera']
    f = float(cam.get('focal_px', 966)); cx, cy = cam.get('principal', [W / 2, H / 2])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    # camera 0: on the normal through the origin, looking along u turned by yaw (to its right) and tilted down by pitch
    yaw, pitch = np.radians(cam.get('yaw_deg', 0.0)), np.radians(cam.get('pitch_deg', 2.0))
    fwd = np.cos(yaw) * u - np.sin(yaw) * v                                  # camera right is -v, so +yaw turns toward -v
    fwd = np.cos(pitch) * fwd - np.sin(pitch) * n
    C0 = o + float(cam.get('height', 0.69)) * n
    w2c0 = rc.look_at(C0, C0 + fwd, n); R0 = w2c0[:, :3]
    # cylinders: positions from keyframes (linear, held at the ends), radius rule, palette by left-to-right order
    cyl = []
    for c in scene['cylinders']:
        kf = sorted(c['keyframes'], key=lambda k: k['frame'])
        fr = np.array([k['frame'] for k in kf], float); ab = np.array([k['uv'] for k in kf], float)
        t = np.arange(NF, dtype=float)
        a, b = np.interp(t, fr, ab[:, 0]), np.interp(t, fr, ab[:, 1])
        pos = o[None] + a[:, None] * u[None] + b[:, None] * v[None]
        h = float(c.get('height', 1.0))
        cyl.append(dict(track_id=c.get('id', len(cyl)), height=h, radius=float(c.get('radius') or 0.2 * h),
                        pos=pos.tolist(), color_index=c.get('color_index')))
    s_grid = float(np.median([c['height'] for c in cyl])) if cyl else 1.0
    allpos = np.array([c['pos'] for c in cyl]).reshape(-1, 3) if cyl else np.zeros((0, 3))
    T = np.array([c['pos'][40] for c in cyl]).mean(0) + 0.5 * s_grid * n if cyl else o + 3 * s_grid * u
    path = cam.get('path', 'static'); assert path in rc.NAMES, f'camera.path must be one of {rc.NAMES}'
    w2cs = rc.camera_paths(path, C0, R0, n, o, u, v, T, s_grid)
    # colours: training assigns the palette by the left-to-right order of the feet in the first frame
    if cyl:
        x0 = project(K, w2cs[0], np.array([c['pos'][0] for c in cyl]))[0][:, 0]
        order = np.argsort(x0, kind='stable')
        for rank, i in enumerate(order):
            ci = cyl[i]['color_index']
            cyl[i]['color'] = PALETTE[(ci if ci is not None else rank) % len(PALETTE)]
    pos_uv = np.stack([(allpos - o) @ u, (allpos - o) @ v], 1) if cyl else np.zeros((0, 2))
    pos_uv = np.vstack([pos_uv, [[(C0 - o) @ u, (C0 - o) @ v]]])          # same extent rule as rerender_camera.main
    return K, w2cs, n, o, u, v, cyl, s_grid, pos_uv


def contact_sheet(frames, path, idx=(0, 20, 40, 60, 80), cell=(416, 234)):
    ims = [frames[k].resize(cell, Image.BILINEAR) for k in idx]
    sheet = Image.new('RGB', (cell[0] * len(ims) + 6 * (len(ims) - 1), cell[1] + 26), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (k, im) in enumerate(zip(idx, ims)):
        x = i * (cell[0] + 6); sheet.paste(im, (x, 26)); d.text((x + 4, 6), f'frame {k}  ({k / rc.FPS:.2f} s)', fill=(0, 0, 0))
    sheet.save(path)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    name = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--name=')), None)
    scene = json.load(open(args[0])); out = args[1]
    name = name or scene.get('name') or os.path.splitext(os.path.basename(args[0]))[0]
    K, w2cs, n, o, u, v, cyl, s_grid, pos_uv = expand_scene(scene)
    frames = rc.render(K, w2cs, n, o, u, v, cyl, s_grid, pos_uv)
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    rc.encode(frames, f'{d}/control.mp4')
    contact_sheet(frames, f'{d}/sheet.png')
    json.dump(dict(K=K.tolist(), w2c=[w.tolist() for w in w2cs], path=scene['camera'].get('path', 'static')), open(f'{d}/cams.json', 'w'))
    json.dump(dict(main_subjects=[c['track_id'] for c in cyl], subjects=cyl), open(f'{d}/cylinders.json', 'w'))
    json.dump(dict(normal=n.tolist(), origin=o.tolist(), u=u.tolist(), v=v.tolist(), grid_spacing=s_grid,
                   camera_height=float((cam_center(w2cs[0]) - o) @ n)), open(f'{d}/plane.json', 'w'), indent=1)
    print(f'{name}: {len(cyl)} cylinders, camera path {scene["camera"].get("path", "static")}, wrote {d}/control.mp4 and sheet.png')


if __name__ == '__main__':
    main()
