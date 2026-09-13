"""Fit the ground of a background photograph before authoring: GeoCalib (single image) gives the focal length and the
gravity direction, i.e. the horizon; we convert them to the editor's camera (focal_px at the scene width, pitch_deg = camera
tilt below the horizon; roll is reported but not used). Camera height stays a user choice in subject-height units.
Usage: .venv/bin/python editor/fit_background.py photo.png [--scene in.json --out out.json] [--width 1280]
Prints focal / pitch / horizon row; with --scene it writes a copy of the scene with camera.focal_px and camera.pitch_deg replaced."""
import sys, json, argparse
import numpy as np, torch
from PIL import Image
from geocalib import GeoCalib

def fit(path, width=1280):
    model = GeoCalib().eval(); img = model.load_image(path)
    with torch.no_grad(): r = model.calibrate(img)
    cam, grav = r['camera'], r['gravity']
    f = float(cam.f.mean()); cx, cy = [float(x) for x in cam.c.squeeze()]
    roll, pitch_gc = [float(np.degrees(x)) for x in grav.rp.squeeze()]
    g = grav.vec3d.squeeze().numpy().astype(float); K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    l = np.linalg.inv(K).T @ g; y_h = float(-(l[0] * cx + l[2]) / l[1])          # horizon row at the centre column
    H, W = img.shape[-2:]; s = width / W
    pitch_down = float(np.degrees(np.arctan2(cy - y_h, f)))                           # + = camera looks down (horizon above centre)
    return dict(image=str(path), image_wh=[W, H], focal_px=f * s, horizon_y=y_h * s, pitch_deg=pitch_down, roll_deg=roll, geocalib_pitch_deg=pitch_gc, scene_width=width)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('photo'); ap.add_argument('--scene'); ap.add_argument('--out'); ap.add_argument('--width', type=int, default=1280)
    a = ap.parse_args(); res = fit(a.photo, a.width)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}))
    if a.scene:
        s = json.load(open(a.scene)); s['camera']['focal_px'] = res['focal_px']; s['camera']['pitch_deg'] = res['pitch_deg']; s['camera']['fitted_from'] = res['image']; s['camera']['anchor_first_frame'] = True; s['camera']['fit'] = res
        json.dump(s, open(a.out or a.scene, 'w'), indent=1); print('wrote', a.out or a.scene)
