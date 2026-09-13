# CoaG authored scene (`scene.json`)

One file describes everything the renderer needs: the ground plane, the cylinders with their positions over time, and the camera. Units are **subject heights**: a cylinder of height 1.0 is one "person", and the grid spacing is the median cylinder height, exactly as in the training data (where the grid spacing was the median subject height).

```json
{
  "version": 1,
  "name": "two_people_orbit",
  "n_frames": 81, "fps": 16, "width": 1280, "height": 720,
  "plane":  {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0], "normal": [0, 0, 1]},
  "cylinders": [
    {"id": 0, "height": 1.0, "radius": 0.2, "color_index": null,
     "keyframes": [{"frame": 0,  "uv": [2.6, 0.5],  "handle_out": [3.2, 0.5]},
                   {"frame": 40, "uv": [3.0, 0.0],  "handle_in": [3.0, 0.3], "handle_out": [3.0, -0.3]},
                   {"frame": 80, "uv": [2.6, -0.5], "handle_in": [3.2, -0.5]}]},
    {"id": 1, "height": 0.9, "keyframes": [{"frame": 0, "uv": [2.6, -0.5]}, {"frame": 80, "uv": [2.6, 0.5]}]}
  ],
  "camera": {"focal_px": 966, "principal": [640, 360], "height": 0.69, "pitch_deg": 2.0, "yaw_deg": 0.0, "path": "orbit"}
}
```

## Frame conventions (identical to `preprocess/step4_render.py` and `rerender_camera.py`)

- The world frame **is** the plane frame: `origin` is the foot of the perpendicular from camera 0 onto the ground, `u` is camera 0's forward direction projected onto the ground, `normal` points up (toward the camera side), `v = normal x u`. With the canonical values above, a point on the ground at plane coordinates `(a, b)` is the world point `origin + a*u + b*v = (a, b, 0)`.
- `u` (first coordinate) is **forward, away from camera 0**; `v` (second coordinate) points to the **camera's left**. So a person at `uv = [2.6, -0.5]` stands 2.6 subject heights in front of the camera and half a height to the right. The editor draws the top view with the camera at the bottom and the right-hand side on the right, and does this flip for you.
- Cameras follow OpenCV: `w2c` is 3x4 world-to-camera, x right, y down, z forward; `K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]` in pixels of a `width x height` image (training: 1280x720, principal point at the centre, f median 966 px, quartiles 865-1082).

## Fields

| field | meaning | default |
| --- | --- | --- |
| `n_frames`, `fps` | fixed by the training data | 81, 16 |
| `width`, `height` | output size of the control video | 1280, 720 (the trainer resizes to its 480p bucket) |
| `plane` | plane frame; leave canonical unless you have a reason | identity as above |
| `cylinders[].height` | height in subject heights | 1.0 |
| `cylinders[].radius` | if omitted, `0.2 * height` (the training rule) | `0.2 * height` |
| `cylinders[].color_index` | index into the 6-colour palette `#E53935 #1E88E5 #43A047 #FDD835 #8E24AA #FB8C00`; **leave null** to get the training rule: colours assigned by left-to-right order of the projected feet in frame 0 | null |
| `cylinders[].keyframes` | list of `{frame, uv, handle_in?, handle_out?}`; the path is a chain of cubic Bezier segments, see below; held constant before the first and after the last keyframe | at least one keyframe |
| `camera.height` | camera 0 height above the ground in subject heights (training median 0.69, quartiles 0.56-0.85) | 0.69 |
| `camera.pitch_deg` | tilt down of camera 0 in degrees (training median 2.2, quartiles -0.6..5.6) | 2 |
| `camera.yaw_deg` | turn of camera 0 to its right, degrees | 0 |
| `camera.focal_px` | focal length in pixels for `width` | 966 |
| `camera.fit` | optional, the JSON printed by `fit_background.py` for the background photo (`focal_px`, `pitch_deg`, `horizon_y`, `roll_deg`, in the 1280x720 frame). Bookkeeping only: the renderer reads `focal_px`/`pitch_deg` from the camera fields above, which the editor sets from the fit on import; the editor draws `horizon_y` as the "photo horizon" line. Kept verbatim across import/export. | absent |
| `camera.anchor_first_frame` | optional boolean, set by `fit_background.py` and the editor checkbox *anchor first frame to the photo*: frame 0 keeps the fitted camera `(C0, R0)` exactly, and the re-aiming presets become rigid moves from there: orbit rotates 0..+40 deg about the vertical through the people's centroid (`R_t = R0 Rm^T`, `C_t = T + Rm (C0 - T)`), pan yaws 0..+24 deg, crane rises 1.5 h with the orientation kept; static and dolly are unchanged | false |
| `camera.background_name` | optional, file name of the background photo loaded in the editor (the image itself is not stored) | absent |
| `camera.path` | `static`, `dolly_in`, `dolly_out`, `orbit`, `pan`, `crane`: the camera moves relative to camera 0 exactly as `rerender_camera.camera_paths` defines them (dolly = min(1.2 h, 40% of the distance to the subjects), orbit = -20..+20 deg around the subjects' centroid, pan = -12..+12 deg yaw, crane = start 1.5 h higher aimed at the centroid and descend) | static |

There is no explicit camera distance: the origin is under camera 0 by convention, so "moving the camera back" is the same as moving every cylinder forward (the editor's *scene shift* buttons do exactly that). In the training data the subjects stood a median 2.6 subject heights in front of the camera at frame 0 (quartiles 2.1-3.4).

## Trajectories: Bezier segments between keyframes

Consecutive keyframes `i -> i+1` are joined by one cubic Bezier with control points `P0 = uv_i`, `P1 = handle_out_i`, `P2 = handle_in_{i+1}`, `P3 = uv_{i+1}`. Handles are **absolute plane coordinates**. A missing handle takes the Catmull-Rom tangent `P_i +/- (P_{i+1} - P_{i-1}) / 6` (endpoints duplicated), so a scene without handles is a smooth spline through its keyframes and a two-keyframe path is a straight line: files written before this field existed render unchanged.

Timing: a person reaches keyframe `i` exactly at `frame_i` and moves at **constant speed along the curve within each segment** (arc length from a 200-sample table, identical in the editor and in `render_authored.py`). Speed is therefore set only by the keyframe frames, which every person chooses independently (the editor's `keyframe frames` field retimes the selected person, or all people with *apply to all people*).

## What the renderer writes (`render_authored.py scene.json out_dir`)

`out_dir/<name>/control.mp4` (the control video), `cams.json` (`K`, per-frame `w2c`, `path`), `cylinders.json` and `plane.json` in the same shape as step 4 writes them, and `sheet.png` (frames 0, 20, 40, 60, 80).
