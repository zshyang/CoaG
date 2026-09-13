"""Visualize Mengyi's cams10 + ground points: fitted plane, 10 camera poses, trajectory. Writes sheet PNGs."""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import fit_ground, cam_center, cam_axes_world, interpolate_cams

root = sys.argv[1]; out = sys.argv[2]; ids = sorted(d for d in os.listdir(root) if d.startswith('c') and len(d) == 5)
os.makedirs(out, exist_ok=True)
rows = []
for cid in ids:
    gp = np.load(f'{root}/{cid}/ground_points.npz'); cams = np.load(f'{root}/{cid}/cams10.npz')
    pts = gp['points'].astype(float); md = float(gp['median_depth'])
    K10, w2c10, kf = cams['K'], cams['w2c'], cams['keyframes']
    g = fit_ground(pts, md, w2c10[0])
    Kall, w2call = interpolate_cams(K10, w2c10, kf)
    C10 = np.stack([cam_center(w) for w in w2c10]); Call = np.stack([cam_center(w) for w in w2call])
    n, o, u, v = g['normal'], g['origin'], g['u'], g['v']
    # plane-frame coords for a top-down view
    to_uv = lambda X: np.stack([(X - o) @ u, (X - o) @ v], -1)
    fig = plt.figure(figsize=(11, 4.6), dpi=110)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    sub = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
    inl = np.abs(sub @ n + g['d']) < g['thr']
    # plane-frame coordinates: X along u, Y along v, Z = height above the plane (so the ground is flat at Z=0)
    to_uvn = lambda X: np.stack([(X - o) @ u, (X - o) @ v, (X - o) @ n], -1)
    P = to_uvn(sub); Cp = to_uvn(C10); Cpath = to_uvn(Call)
    ax.scatter(P[inl, 0], P[inl, 1], P[inl, 2], s=1, c='#43A047', alpha=0.5)
    ax.scatter(P[~inl, 0], P[~inl, 1], P[~inl, 2], s=1, c='#bbbbbb', alpha=0.4)
    scale = 0.15 * md
    for i, w in enumerate(w2c10):
        f, up, r = cam_axes_world(w)
        fp = np.array([f @ u, f @ v, f @ n]); upp = np.array([up @ u, up @ v, up @ n])
        ax.quiver(*Cp[i], *(fp * scale), color='#E53935', arrow_length_ratio=0.25)
        ax.quiver(*Cp[i], *(upp * scale * 0.6), color='#1E88E5', arrow_length_ratio=0.3)
    ax.plot(Cpath[:, 0], Cpath[:, 1], Cpath[:, 2], color='#E53935', lw=1)
    ax.scatter(Cp[:, 0], Cp[:, 1], Cp[:, 2], c='k', s=14)
    allp = np.vstack([P, Cp]); lo, hi = allp.min(0), allp.max(0); ctr = (lo + hi) / 2; rad = (hi - lo).max() / 2
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad); ax.set_zlim(min(-0.05 * rad, lo[2]), ctr[2] + rad)
    ax.view_init(elev=22, azim=-135)
    ax.set_xlabel('u (fwd)'); ax.set_ylabel('v'); ax.set_zlabel('height'); ax.tick_params(labelsize=6)
    ax.set_title(f'{cid}: inliers {g["inlier_ratio"]:.2f}, normal∠up {g["normal_angle_deg"]:.0f}°, cam height {g["camera_height"]:.2f}, {"OK" if g["plane_ok"] else "FAIL"}\nred = camera forward, blue = camera up, black = 10 keyframe cameras', fontsize=8)
    # top-down (plane frame)
    ax2 = fig.add_subplot(1, 2, 2)
    puv = to_uv(sub); ax2.scatter(puv[inl, 0], puv[inl, 1], s=1, c='#43A047', alpha=0.5); ax2.scatter(puv[~inl, 0], puv[~inl, 1], s=1, c='#bbbbbb', alpha=0.3)
    cuv = to_uv(Call); ax2.plot(cuv[:, 0], cuv[:, 1], color='#E53935', lw=1.2)
    c10uv = to_uv(C10); ax2.scatter(c10uv[:, 0], c10uv[:, 1], c='k', s=14, zorder=3)
    for i, w in enumerate(w2c10):
        f, _, _ = cam_axes_world(w); fu = np.array([f @ u, f @ v]); fu = fu / (np.linalg.norm(fu) + 1e-9) * 0.2 * md
        ax2.annotate('', xy=c10uv[i] + fu, xytext=c10uv[i], arrowprops=dict(arrowstyle='->', color='#E53935', lw=1))
    ax2.set_aspect('equal'); ax2.set_title('top-down in plane frame (u = camera forward, v = left); red = camera path', fontsize=9)
    ax2.set_xlabel('u'); ax2.set_ylabel('v'); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{out}/cams_{cid}.png'); plt.close(fig)
    path_len = float(np.linalg.norm(np.diff(Call, axis=0), axis=1).sum())
    rows.append((cid, g['inlier_ratio'], g['normal_angle_deg'], g['camera_height'], path_len / md, g['plane_ok']))
print('id      inlier  normal∠up  cam_h   path/medD  plane_ok')
for r in rows: print('%s   %.2f    %5.1f°    %.2f    %.2f      %s' % r)
