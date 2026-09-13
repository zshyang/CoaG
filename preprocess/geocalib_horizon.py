"""Single-image horizon (gravity) from GeoCalib on frame 0 of each clip; CPU.
Usage: python geocalib_horizon.py <clip81_dir> <out_json> [ids...]   (default ids: all c????.mp4 in clip81_dir; skips ids already in out_json)
Output per id: gravity_cam (unit, points down, camera coords: x right, y down, z forward), roll_deg, pitch_deg,
focal_gc (GeoCalib's own focal, px), horizon_y_center (pixel row of the horizon at the image centre column, 1280x720)."""
import sys, os, json, subprocess, tempfile, time
import numpy as np, torch
from geocalib import GeoCalib

def main():
    clip_dir, out = sys.argv[1], sys.argv[2]
    ids = sys.argv[3:] or sorted(f[:-4] for f in os.listdir(clip_dir) if f.endswith('.mp4') and f.startswith('c') and len(f) == 9)
    res = json.load(open(out)) if os.path.exists(out) else {}
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model = GeoCalib().eval()
    tmp = tempfile.mkdtemp()
    t0 = time.time(); n = 0
    for cid in ids:
        if cid in res:
            continue
        png = f'{tmp}/{cid}.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{clip_dir}/{cid}.mp4', '-frames:v', '1', png], check=True)
        img = model.load_image(png)
        with torch.no_grad():
            r = model.calibrate(img)
        cam, grav = r['camera'], r['gravity']
        f = float(cam.f.mean()); cx, cy = [float(x) for x in cam.c.squeeze()]
        roll, pitch = [float(np.degrees(x)) for x in grav.rp.squeeze()]
        g = grav.vec3d.squeeze().numpy().astype(float)          # gravity direction in camera coords
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        l = np.linalg.inv(K).T @ g                               # horizon line
        y_c = float(-(l[0] * cx + l[2]) / l[1]) if abs(l[1]) > 1e-9 else None
        H, W = img.shape[-2:]
        sx, sy = 1280 / W, 720 / H                                # in case load_image resized
        res[cid] = dict(gravity_cam=g.tolist(), roll_deg=roll, pitch_deg=pitch, focal_gc=f * sx, cx=cx * sx, cy=cy * sy,
                        horizon_y_center=(y_c * sy if y_c is not None else None), img_wh=[W, H])
        n += 1
        if n % 20 == 0 or n == len(ids):
            json.dump(res, open(out, 'w'), indent=1)
            print(f'{n} done  {(time.time() - t0) / n:.1f} s/img', flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    for cid in ids:
        r = res[cid]; print(cid, 'horizon_y %.0f' % r['horizon_y_center'], 'pitch %.1f' % r['pitch_deg'], 'roll %.1f' % r['roll_deg'], 'f %.0f' % r['focal_gc'])

if __name__ == '__main__':
    main()
