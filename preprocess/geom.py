"""Shared geometry for step 4: plane fit, camera helpers, interpolation. numpy + scipy only."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def cam_center(w2c):
    R, t = w2c[:, :3], w2c[:, 3]
    return -R.T @ t

def cam_axes_world(w2c):
    """Returns (forward, up, right) unit vectors in world coords for an OpenCV camera (z forward, y down)."""
    R = w2c[:, :3]
    return R.T @ np.array([0, 0, 1.0]), R.T @ np.array([0, -1.0, 0]), R.T @ np.array([1.0, 0, 0])

def fit_plane_ransac(pts, thr, iters=600, seed=0, normals=None, max_normal_deg=30.0):
    """RANSAC plane. With per-point normals (WorldMirror), half the hypotheses are 1-point (point + its normal)
    and inliers must also agree with the plane normal within max_normal_deg; without normals, plain 3-point RANSAC."""
    rng = np.random.default_rng(seed)
    cos_min = np.cos(np.radians(max_normal_deg))
    best_n, best_d, best_in = None, None, np.zeros(len(pts), bool)

    def inliers(n, d):
        inl = np.abs(pts @ n + d) < thr
        if normals is not None:
            inl &= np.abs(normals @ n) > cos_min
        return inl

    for it in range(iters):
        if normals is not None and it % 2 == 0:
            i = rng.integers(len(pts))
            n = normals[i].copy()
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n /= nn
            d = -n @ pts[i]
        else:
            i = rng.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[i]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            d = -n @ p0
        inl = inliers(n, d)
        if inl.sum() > best_in.sum():
            best_n, best_d, best_in = n, d, inl
    if best_n is None:
        raise RuntimeError('RANSAC found no plane')
    # least-squares refine on inliers
    q = pts[best_in]
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n = vt[-1]
    n /= np.linalg.norm(n)
    d = -n @ c
    inl = inliers(n, d)
    return n, d, inl

def plane_frame(n, d, w2c0):
    """Origin = foot of the perpendicular from camera 0; u = camera forward projected onto the plane; v = n x u."""
    C = cam_center(w2c0)
    h = n @ C + d
    origin = C - h * n
    fwd, _, _ = cam_axes_world(w2c0)
    u = fwd - (fwd @ n) * n
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0, 0]) - n[0] * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return origin, u, v, h

def fit_ground(points, median_depth, w2c0, normals=None):
    thr = 0.01 * float(median_depth)
    n, d, inl = fit_plane_ransac(points, thr, normals=normals)
    C = cam_center(w2c0)
    if n @ C + d < 0:          # orient the normal toward the camera side
        n, d = -n, -d
    _, up, _ = cam_axes_world(w2c0)
    angle = float(np.degrees(np.arccos(np.clip(n @ up, -1, 1))))
    origin, u, v, h = plane_frame(n, d, w2c0)
    agree = float(np.mean(np.abs(normals[inl] @ n))) if normals is not None and inl.any() else None
    return dict(normal=n, d=float(d), inliers=inl, inlier_ratio=float(inl.mean()), normal_angle_deg=angle,
                camera_height=float(h), origin=origin, u=u, v=v, thr=thr, normal_agreement=agree,
                plane_ok=bool(inl.mean() >= 0.30 and angle <= 45 and h > 0))

def horizon_y(K, w2c, n, x=None):
    """Pixel y of the plane's vanishing line at image column x (default: principal point). None if the horizon is
    not in front of the camera (e.g. camera looking straight down)."""
    R = w2c[:, :3]
    # vanishing line l = K^-T R n (line of points at infinity in the plane)
    l = np.linalg.inv(K).T @ (R @ n)
    x = K[0, 2] if x is None else x
    if abs(l[1]) < 1e-9:
        return None
    return float(-(l[0] * x + l[2]) / l[1])

def interpolate_cams(K10, w2c10, keyframes, n_frames=81):
    """Slerp rotations / lerp centers between keyframes; returns K[n], w2c[n]."""
    keyframes = np.asarray(keyframes)
    Rs = Rotation.from_matrix(np.stack([w[:, :3] for w in w2c10]))
    Cs = np.stack([cam_center(w) for w in w2c10])
    slerp = Slerp(keyframes.astype(float), Rs)
    ts = np.clip(np.arange(n_frames, dtype=float), keyframes.min(), keyframes.max())
    R_all = slerp(ts).as_matrix()
    C_all = np.stack([np.interp(ts, keyframes, Cs[:, j]) for j in range(3)], 1)
    w2c = np.zeros((n_frames, 3, 4))
    w2c[:, :, :3] = R_all
    w2c[:, :, 3] = -np.einsum('nij,nj->ni', R_all, C_all)
    K = np.stack([K10[int(np.argmin(np.abs(keyframes - t)))] for t in ts])
    return K, w2c

def project(K, w2c, X):
    """X [...,3] world -> pixel [...,2] and depth [...]."""
    Xc = X @ w2c[:, :3].T + w2c[:, 3]
    z = Xc[..., 2]
    uv = (Xc @ K.T)
    return uv[..., :2] / np.maximum(uv[..., 2:3], 1e-9), z

def pixel_ray(K, w2c, u, v):
    d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    d_w = w2c[:, :3].T @ d_cam
    return cam_center(w2c), d_w / np.linalg.norm(d_w)

def ray_plane(C, dirw, n, d):
    denom = n @ dirw
    if abs(denom) < 1e-9:
        return None
    s = -(n @ C + d) / denom
    return None if s <= 0 else C + s * dirw
