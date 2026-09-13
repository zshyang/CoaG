# CoaG authoring editor

Draw the scene by hand, export `scene.json`, render an 81-frame control video that follows the exact conventions of the training data, then run the LoRA on it.

```
editor/
  index.html           the editor: open it in a browser, no server needed
  render_authored.py   scene.json -> control.mp4 (+ cams.json, cylinders.json, plane.json, sheet.png)
  scene_schema.md      the JSON format and the frame conventions
  examples/            two hand-written scenes and their rendered outputs (examples/out/)
```

## 1. Author

Open `editor/index.html` (double-click; Chrome, Safari and Firefox all work, nothing is downloaded).

- **Top view** (left): the camera sits at the bottom looking up the screen; one grid cell is one subject height. Drag a big circle to move that person **at the selected keyframe** (frames 0, 20, 40, 60, 80 by default; each person has its own keyframe frames, edit the *keyframe frames* field to retime the selected person, or tick *apply to all people*). Paths are Bezier curves through the keyframes: the selected person shows small handle circles, drag them to bend the path (hollow = automatic Catmull-Rom tangent, filled = your own; *reset handles* clears them). People move at constant speed along the curve between consecutive keyframes. Tick *move all keyframes together* to translate a whole path. *Add person* / *Remove selected* (1-6 people; colours follow the training rule, left-to-right in frame 0). *closer / farther* shift the whole scene along the view direction, which is how you set the camera distance (the ground origin is always under camera 0, as in the training data).
- **Camera preview** (right): the frames the renderer will produce, drawn with the same projection at half size, with each person's ground path as a dotted line. Scrub or play the 81 frames. Choose a **path** (static, dolly in, dolly out, orbit, pan, crane, with the same magnitudes as the paper's camera experiments) and set camera height, pitch, yaw and focal length. The defaults are the training medians.
- **Export scene.json** (handles are written only where you set them). *import* loads a scene back exactly, including per-person keyframe frames and handles.

## 2. Render

```bash
cd code/wan_control_demo
.venv/bin/python editor/render_authored.py ~/Downloads/my_scene.json runtime/authored/
# -> runtime/authored/my_scene/{control.mp4, sheet.png, cams.json, cylinders.json, plane.json}
```

`render_authored.py` imports the drawing, camera-path and projection functions from `preprocess/rerender_camera.py`, `preprocess/geom.py` and `preprocess/step4_render.py` (nothing is re-implemented), so the video is 1280x720, 16 fps, 81 frames, black background, white grid with spacing = median cylinder height, cylinders of radius 0.2 x height in the training palette, painter's ordering. Look at `sheet.png` before spending GPU time. The interpreter needs numpy, scipy and Pillow (`.venv` has them; `runtime/diarization/venv` works too).

Examples, already rendered:

```bash
.venv/bin/python editor/render_authored.py editor/examples/two_people_orbit.json editor/examples/out
.venv/bin/python editor/render_authored.py editor/examples/four_people_dolly_in.json editor/examples/out
```

## 3. Generate a video with the LoRA

On the inference pod (see `train/README.md`), copy the control video over and pick a background reference image and a prompt that agree with the layout (same number of people, a ground the reference image shows):

```bash
scp -P <port> runtime/authored/my_scene/control.mp4 root@<ip>:/workspace/data/infer_inputs/authored/control_my_scene.mp4
# on the pod: run_infer_case.sh expects CID/CAM paths, so either
CID=authored CAM=my_scene REF=/workspace/data/train_data/ref/c0599.png PROMPT="..." \
  LORA_LOW=/workspace/out/full_low_t640/checkpoint-241.safetensors LORA_HIGH=/workspace/out/full_high_t640/checkpoint-241.safetensors \
  TAG=my_scene /workspace/wan_train/run_infer_case.sh
# or call the env-driven script directly
CONTROL_VIDEO=/workspace/data/infer_inputs/authored/control_my_scene.mp4 REF_IMAGE=/workspace/data/train_data/ref/c0599.png \
  PROMPT="..." LORA_LOW=... LORA_HIGH=... SAVE_PATH=/workspace/out/samples/my_scene \
  MODEL_NAME=/workspace/models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control python examples/wan2.2_fun/infer_control_ref.py
```

(`run_infer_case.sh` reads the prompt from `holdout_captions.json` when `PROMPT` is not given; for an authored scene always pass `PROMPT`.)

## Using a photograph as background

At inference the reference image sets the background, so the authored camera must agree with the photo's ground: same focal length and tilt, otherwise the people come out at the wrong scale or float. Workflow:

1. `python editor/fit_background.py photo.jpg` (GeoCalib; see the script header for options) prints a JSON with `focal_px`, `pitch_deg` (tilt below the horizon), `horizon_y` and `roll_deg`, all expressed in the 1280x720 control-video frame. It can also patch a `scene.json` directly.
2. In the editor: *import camera fit* (sets focal and pitch, height untouched, and stores the fit in `scene.camera.fit`), then *background image* to draw the photo under the preview (object-fit contain, grid drawn semi-transparent on top). A yellow dashed line marks the photo horizon from the fit and a blue dotted line the grid's own vanishing line: they should coincide.
3. Adjust **camera height** until the cylinders' feet sit where people would stand in the photo (a taller camera pushes the ground down in the frame), and move the people with the top view. Use the same photo as `REF_IMAGE` when generating.

Roll is reported but ignored (the renderer has no roll; pick photos with a level horizon). The fitted pitch is kept by the static, dolly and pan paths; orbit and crane re-aim the camera at the people, so their horizon differs from the photo's by design. Only the file name is exported (`camera.background_name`), not the image.

## Conventions worth knowing

- Units are subject heights; the grid spacing is the median cylinder height, so with everyone at 1.0 the grid is one person tall per cell. Keep the people 2-4 units in front of the camera and the camera 0.5-0.9 units high to stay inside the training distribution.
- The second plane coordinate `v` points to the camera's **left** (`v = normal x forward`); the editor shows right on the right and does the flip. See `scene_schema.md`.
- Colours are assigned by left-to-right order of the feet in frame 0, as in training; set `color_index` in the JSON only if you know why.
- The preview in the browser is a JavaScript port of the same math, including the Bezier sampler (checked against the Python renderer over all 81 frames to below 1e-9 px); the Python output is what you feed the model.
