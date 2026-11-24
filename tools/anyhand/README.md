# Frame camera simulation and AnyHand rendering

These are offline research tools. They consume saved SteamOS calibration and
AnyHand mesh annotations as external inputs.

## Camera model

Use the unit's `/persist/xrservice.json` and `/persist/device_config.json`.
`frame_camera.py` uses the pipeline's existing head-from-calibration transform
to export optical centers, axes, native sensor extents, KB4 coefficients and
the full factory KB-spline tables. The renderer uses KB4, the lens model
implemented by the pipeline. It restricts the inverse map to the declared
image circle. The spline data is preserved for later qualification.

The Valve [Steam Frame CAD release](https://gitlab.steamos.cloud/SteamHardware/SteamFrame)
is a mechanical reference. The core module archive at revision `bdbb41ba`
contains one stripped STEP model. The experimental rig uses measured SteamOS
extrinsics and intrinsics instead of inferring optics from the enclosure.

## AnyHand geometry

The upstream [AnyHand generator](https://github.com/chen-si-cs/AnyHand-Generation)
at `ed9dd7b4` samples SAPIEN cameras on an orbit in
`sync_data_utils.py::run_simulation_sapien` and uses a fixed pinhole field of
view. Width and height settings alone cannot describe the Frame rig. Its full
asset-based generator also needs separately supplied MANO, pose, and texture
assets.

The pilot consumes released posed meshes from the
[AnyHand dataset](https://huggingface.co/datasets/chen-si-02/AnyHand-Dataset)
at `e8279bc6`. Its sampled `data/test/single-*` shards include `verts_3d`;
the checked training shards contain joints and pose parameters but omit the
posed vertices. Use disjoint source shards for the custom training and
validation sets. These experiments do not use the AnyHand benchmark protocol.
To use fresh upstream-generated motions, prepare the upstream assets, keep its
mesh metadata, and use its `convert_h5_to_tar.py` path that emits
`verts_3d`, joints, source images and masks. The Frame adapter consumes that
WebDataset contract.

For topology, provide a 778-vertex, 1538-face right-hand OBJ in MANO vertex
order. The pilot used the public topology correspondence
`assets/MANO/base_hand_right.obj` in [SOMA-X](https://github.com/NVlabs/SOMA-X).
The licensed MANO model binary is unnecessary for replaying the released posed
vertices. Keep dataset, model, and asset terms with their respective inputs.

The CPU rasterizer samples pinhole cubemap faces using calibrated inverse KB4
rays and uses a shared depth buffer per face for hand visibility. It supplies
approximate sleeves, grayscale lighting, vignetting, sensor noise and blur.
The rig geometry is calibrated. The material, lighting and camera response are
approximations.

Use Python with NumPy 2, SciPy and OpenCV. Use the same calibration files for
every dataset split.

```sh
c++ -O3 -shared -fPIC raster.cpp -o /tmp/anyhand-raster.so

python frame_camera.py --xrservice xrservice.json \
  --device-config device_config.json --out frame-rig.json
```

## Calibrated stereo dataset

`render_stereo.py` projects the AnyHand scene into a supplied left/right
camera pair. For a rig JSON, set `stream_order` to the two camera keys in
left-to-right order. Each entry under `cameras` supplies its
`source_calibration` and a rigid `pair_from_eye` transform in metres.
The renderer supports KB4 and FISHEYE624 calibrations. It also reads
per-frame HOT3D `*.cameras.json` files with labeled left and right tracking
views. Sensor dimensions, intrinsics, and poses come from the calibration.
The [hand-tracking toolkit](https://github.com/facebookresearch/hand_tracking_toolkit)
provides the FISHEYE624 forward projection.

```sh
python render_stereo.py --calibration 000000.cameras.json \
  --toolkit hand_tracking_toolkit_repo --topology base_hand_right.obj \
  --library /tmp/anyhand-raster.so --source source-a.tar source-b.tar \
  --out stereo-train --seed 501 --count 192
```

Every view has a native-resolution grayscale PNG and a hand mask with values 0 for
background, 1 for left, and 2 for right. `dataset.json` includes visible
boxes, camera-pixel joints, and shared 3D joints in the camera-pair frame.

## Fully synthetic calibrated PBR scenes

`handspotter-render` renders the calibrated stereo pair or four-camera Frame rig in one headless Chromium
process. The three.js scene contains the AnyHand geometry, articulated arms,
and one of three furnished environments. Both arms share the same shoulder
width and segment lengths. Two-segment inverse kinematics places the elbow
for each reach. The hand follows the forearm, with forearm twist and coupled
wrist bend sampled around that frame. The arm surface blends from the
original wrist boundary into tapered elliptical sections. The centerline
has a rounded elbow, and the cross-sections gradually follow wrist twist
toward the hand. Forearm width and depth use the pronation and supination
fits from six scanned subjects, scaled to adult circumference measurements.
The upper-arm and elbow geometry use procedural profiles. The measurements
provide a scale for synthetic adult arms. [Perry et al., Table 3](https://doi.org/10.3389/frobt.2021.576783) · [ANSUR II, measurements 9 and 38](https://dacowits.defense.gov/LinkClick.aspx?fileticket=EbsKcm6A10U%3D&portalid=48)

The clothed torso shares the arm rig's shoulder joints, garment color, and
fabric pattern. The shoulder openings join the upper-arm boundaries and
share their normals. The torso is present on frames with zero, one, or two
visible hands, casts shadows, and occludes hand pixels in the visibility pass.
All surfaces share the same camera pair, visibility buffers, room pose,
and light sources. The scene uses HDR environment lighting, broad fill
from a rectangular emitter, and a shadow-casting key. Window, ceiling, and
task-light setups vary with the seed, as do exposure, room orientation,
skin tone, cuffs, and prop placement. The skin combines tile-based color,
roughness, and small relief with mesh-relative palms, creases, knuckles,
and nails. Sleeves use a textile texture, and environment geometry can
occlude the hands. The renderer reads back linear luminance and applies
exposure, vignetting, and noise before encoding the grayscale output.

The Nix shell pins three.js and the material assets by content hash and
uses Node and Chromium from the flake's `nixpkgs`. Rendering needs a
working hardware Vulkan driver that Chromium's ANGLE backend can use.
Supply the calibration, toolkit checkout, topology, and AnyHand shards as
for the CPU generator.

```sh
nix run .#handspotter-render -- \
  --calibration 000000.cameras.json --toolkit hand_tracking_toolkit_repo \
  --topology base_hand_right.obj --source source-a.tar source-b.tar \
  --seed 501 --count 36 --out stereo-pbr
```

`--environment desk`, `--environment workbench`, and `--environment kitchen`
select a single layout. The default mixes the three within the dataset.
The output includes the same grayscale and mask PNGs, visible boxes,
and projected joints as the CPU generator. With `--rig-json`, it also
writes the Handspotter input tensors to `handspotter.npy`. The manifest
also records the environment and lighting for every stereo pair, plus the
shoulder, elbow, wrist, and sampled joint angles for each arm.

The skin detail uses a generic procedural material with anatomical
regions. The environments are built from 3D geometry and textured surfaces.
The dataset contains no photographed backplates. [TextureCan skin material](https://www.texturecan.com/details/574/) · [Poly Haven lighting and surfaces](https://polyhaven.com/)

Use `--appearance-variation` for a broader skin palette, variable palm,
finger, wrist, and arm proportions, and a mix of bare arms, jersey,
plaid, knit, and athletic sleeves. Both arms share the sampled body and
garment. The renderer uses the deformed hand mesh for projection and
deforms its joint labels with it. Color is converted to luminance for the
monochrome cameras.

Use `--capture-noise` for a mixture of clean and hazy lenses, crossed-arm
reaches, and sparse airborne fragments. Haze scatters light in each eye
with a shared severity and separate smudge positions. The two cameras
see the same 3D occluders from their calibrated viewpoints. The visibility
pass excludes hand pixels hidden by arms or fragments. The manifest
records the sampled appearance and capture parameters. Both flags can
be used independently or together.

The PBR renderer takes a separate flag for each capture effect.

- `--slight-blur` samples mild optical softness per view. The Gaussian
  blur is applied in linear light before sensor noise. The manifest
  records its standard deviation in native pixels.
- `--lens-decentering` samples a fixed optical-center shift and a small
  tangential residual per camera. The shifted projection drives rays,
  masks and landmarks. The generated `rig.json` records the residual
  under each camera's `lens_decentering` alongside its nominal calibration.
- `--arm-motion-blur` samples motion independently for 12% of posed arms,
  with 0.4° to 1.8° of rotation during the exposure. The shoulder-driven
  velocity tapers into the torso. Baked arms and fitted sleeves share
  the same velocity field.
- `--camera-motion-blur` moves all cameras as a rigid headset in 12% of
  frames. It samples 0.12° to 0.45° of rotation and translation of up to
  1 mm per axis during the exposure.

Motion is applied as a screen-space filter in the calibrated fisheye window
using one beauty render and a velocity pass. Masks, bounding boxes and
keypoints describe the middle of the exposure. Pixels newly revealed during
motion are approximated from that view. The dataset records the sampled
motion in each affected scene's `shutter_motion`.

### Torso-relative reaches and head turns

- `--hemisphere-reach` draws wrist targets across the volume in front of
  the torso. Each arm keeps its segment lengths and elbow limits, and the
  sampler uses limb clearance estimates to avoid intersecting the torso or
  opposite arm. With this flag, `--capture-noise` supplies haze and debris.
- `--head-look` samples a separate head pose up to 85° either side and from
  76° down to 70° up, with a small roll. For scenes with hands, 80% of
  draws are independent and 20% are biased toward one hand. There is no
  visibility requirement, so some cameras see no hands.

Pass both flags to create a new template for SOMA-X or MPFB. The bakers
recover the torso-relative hand targets from the template and preserve the
chosen head direction. They scale shoulder-relative reaches by each body's
arm length. Use `--head-look` with an existing body bank to add
head turns to its existing arm poses. The baked skin's connected head and neck
follow the rig while retaining the arm pose. All cameras keep their calibrated
relative positions. The manifest's `head_pose.reference_from_pair` maps the
face-relative 3D hand labels back to the straight-ahead reference where the
room and body are placed.

Export an interactive inspection view of the mount from a SOMA-X body bank
baked with `--cloth-reference` and matching Frame calibrations. The HTML bundles the pinned
Three.js build, the cropped body meshes and camera geometry for offline use.
The OpenVR toggle preserves the front-camera midpoint so it isolates the
axis convention from the assumed fit to the face. The head deformation
toggle compares the renderer's posed-mesh selection with a head region
selected in the neutral mesh.

```sh
nix develop "path:.#anyhand-pbr" --command python \
  tools/anyhand/export_head_rig_viewer.py \
  --body-bank soma-body --rig-json frame-handspotter-rig.json \
  --openvr-calibration frame-rig.json --scene 00000 --out frame-camera-mount.html
```

Use the OpenVR calibration exported by `frame_camera.py` and repeat
`--scene` to inspect additional records.

### Live SOMA-X and AnyHand Studio

Launch `nix run .#anyhand-studio` on x86-64 Linux to open **Batch renderer**.
The packaged starter job opens with a four-camera Steam Frame reference rig,
its matching OpenVR mount, authored reach poses, SOMA-X assets and correctives,
a garment library and nine Foundation rooms. **Render batch** writes the
dataset to a new directory. Select a sample to inspect its camera images,
masks or detector inputs.

For a headless render using only packaged assets:

```sh
nix run .#anyhand-studio -- batch --out ./starter-render
nix run .#handspotter-render -- --environment foundation-classroom --out ./classroom-render
```

| Flake package | Contents |
| --- | --- |
| `anyhand-render-project` | Complete data-only job at `share/anyhand/render-job.json` |
| `handspotter-reference-calibration` | Matched `rig.json` and `openvr.json` under `share/handspotter/calibration` |
| `anyhand-starter-poses` | Forward reach, raised hand and rest under `share/anyhand/templates/starter.json` |
| `anyhand-foundation-scenes` | glTF files, buffers, textures and upstream room credits under `share/anyhand/foundation-scenes` |
| `anyhand-soma-assets`, `anyhand-garment-library` | SOMA-X model, corrective checkpoint and compiled garment patterns |

The reference calibration is for synthetic rendering. For alignment with
recordings from another headset, use that headset's own matched calibration.

The default batch generates new reaches, gestures, head poses, identities, outfits,
rooms, lighting and capture effects for each sample. The template supplies hand
landmarks, so the batch can exceed its row count. Scheduled negatives alternate
between resting wearers and empty rooms. Lens offsets are shared by the samples
from one simulated headset. Clear **Generate new reaches per sample** to replay
the template rows instead.

Studio picks a fresh batch seed on launch. **Reroll** picks another seed and
updates the selected sample. Rendering or saving uses the displayed seed.
The CLI also picks a fresh seed when omitted and records it in `dataset.json`.
Use `--seed 42 --count 24` to reproduce a 24-sample run.

**Save job** writes the settings as JSON. Replay them headlessly with the
same renderer, assets and postprocessing used by the GUI:

```sh
nix run .#anyhand-studio -- batch --job render-job.json --out ./training-render
nix run .#anyhand-studio -- batch --help
```

The batch command also accepts the existing `handspotter render` options.
Relative input paths in a job resolve beside the job file. Choose a fresh
output directory for each run.

Normal batch exports write `coco.json` beside `dataset.json`. Use the batch
output directory as the COCO image root. Every rendered camera view has an
image entry, including views with no visible hands. The `left_hand` and
`right_hand` categories use COCO boxes in `[x, y, width, height]` order,
compressed RLE instance masks, visible mask area in pixels, and a 21-point hand
skeleton. A projected keypoint in front of the camera and inside the image has
visibility `2` if it lands on its own hand mask, and `1` otherwise.
Out-of-frame or behind-camera landmarks have visibility `0`.
`dataset.json` keeps the camera grouping and 3D hand landmarks, and `rig.json`
keeps the calibration. Use `--tensors-only` to skip PNGs and the COCO file
while retaining the native manifests and detector tensors.

**Environment** defaults to the recorded room for template and baked-body jobs.
New jobs use **random** to sample both procedural and packaged Foundation rooms. Choose a named room to pin it in the viewport and the export. The default wardrobe samples built-in styles and the compiled garment library. Every added garment is draped on the posed body with WebGPU. **Garment** lists the built-in
styles and patterns in the selected garment library. Choose **No added garment**
to use the body as supplied.

The viewport identifies the resolved room, its source, and the added garment for
the displayed sample. Each exported dataset record includes the same decisions
in `render_setup`, with the original sample index and cloth mode.

The live viewport keeps the renderer, camera and SOMA model loaded across edits.
Changing the room replaces the environment and lighting while retaining the posed body and garment.
Drag to orbit around the wearer with world vertical fixed, or scroll to zoom. Select a rig camera to follow its calibrated optical axis as the head turns.
Switch to **Detector inputs** to see the calibrated, postprocessed views for that
sample before export. The 3D scene responds to camera movement interactively.
Detector mode captures the posed scene in the same Chromium viewport. Studio and batch use the same WebGL sensor passes for lens inversion, cubemap projection, and motion blur. A reroll sends the new calibration coefficients to Chromium. The GPU applies lens haze, optical softness, seeded shot noise, grayscale encoding, and detector resampling. Studio reads back the detector bytes and hand bounds. Batch also reads the native image and mask when exporting PNGs. Noise is repeatable for a given seed, sample, and camera. Its GPU generator produces different noise samples from older NumPy-based exports.

For a custom dataset or Frame camera calibration, put a project file at
`${XDG_CONFIG_HOME:-$HOME/.config}/anyhand-studio/project.json`. Use the batch form to select replacement files after a move. The pinned model assets can also be
used by custom projects through the `anyhand-soma-assets` flake package.

```json
{
  "assets": "/path/to/SOMA-assets",
  "correctives": "/path/to/correctives_model.pt",
  "template": "/path/to/template/dataset.json",
  "rig": "/path/to/frame-handspotter-rig.json",
  "openvr": "/path/to/frame-rig.json"
}
```

Relative paths are resolved from the project file. Use the optional `garments`
directory for a custom GarmentCode library. Set the project in Nix with
`pkgs.anyhand-studio.override { projectFile = yourProjectFile; }`.

The pose evaluator uses SOMA-X's `Neck1`, `Neck2`, and `Head` joints. Frame's
neutral camera orientations come from the calibration in OpenVR head axes.
The front-pair seating still uses the modeled 95 mm eye-to-camera depth.
The native path uses stock SOMA-X arm skinning and retargets each elbow in
its signed native flexion direction. It checks the posed forearms and hands
against the head after applying the requested head pose. When they intersect,
it tries arm joint adjustments and checks the re-skinned result. Meshes and
hand labels follow the adjusted pose. The head and camera rig keep their
requested pose.

The batch path uses the same worker with the existing headless Chromium
renderer. For a replay, pass the same rendered template to the native worker
through the batch command.

```sh
worker=$(nix build "path:.#anyhand-soma-worker" --no-link --print-out-paths)
nix run .#handspotter-render -- \
  --toolkit hand_tracking_toolkit_repo --rig-json frame-handspotter-rig.json \
  --openvr-calibration frame-rig.json --soma-template template/dataset.json \
  --soma-worker "$worker/bin/anyhand-soma-worker" \
  --soma-assets SOMA-assets --soma-correctives correctives_model.pt \
  --seed 501 --soma-seed 681721 --count 24 --out frame-live
```

For new reaches in a single command, replace `--soma-template` with
`--source`, `--topology`, and the desired reach options. `--hemisphere-reach`
and `--head-look` work with generated targets. Both paths stage resolved
meshes in the renderer's temporary directory and skip the body-bank bake.
Use the same SOMA seed in Studio and batch rendering to match identities.

New native datasets default to `--soma-hand-poses mixed`. Half of each
complete sampling block uses the original SOMA-X finger prior. The other half
is split equally among open hands, fists, pointing, and pinches to the index,
middle, ring, and little finger. Left and right are shuffled separately.
Flexion and spread vary within a gesture. Pinches fit the posed digits to
the wearer's fingertip geometry with a sampled gap for close and approaching
pinches. The manifest records each hand's gesture and measured fingertip
clearance for pinches. Use `--soma-hand-poses gestures` to sample only the
named gestures, or `--soma-hand-poses prior` for the original sampler.
Template replay defaults to the original sampler.


### Foundation-Resources environments

Nine environments can replace the procedural layouts. The classroom, dining
room, kitchen, bathroom, bedroom, and two living rooms place the hands
near furniture or a work surface. The staircase and the Sponza courtyard
use standing viewpoints in open space, with daylight in the courtyard.
The imported geometry and materials participate in the calibrated stereo
render, including occlusion and shadows. The renderer supplies the
cameras and lighting.
The Nix package pins each scene asset by SHA-256 and includes the original room credits.
[Foundation-Resources](https://github.com/mos9527/Foundation-Resources) · [Original indoor scene credits](https://benedikt-bitterli.me/resources/) · [Sponza credits](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza)

For a separate working copy outside the Nix package:

```sh
python3 tools/anyhand/foundation_scenes.py foundation-scenes
```

Add `--foundation-scenes foundation-scenes` to the PBR render command to
mix the supported scenes present in that directory. Use `--scene` with
the downloader to fetch one scene, or repeat it to fetch a subset. To
render a single room, use `--environment foundation-<scene>`, for example
`--environment foundation-living-room-2`. The dataset manifest records
the viewpoint placement and file digests.

### Four-camera Handspotter geometry sweep

`handspotter_camera_sweep.py rigs` builds five four-camera configurations between
`--source-front` and the calibrated Steam Frame rig. It aligns each rig to
its front-camera midpoint, baseline, and optical-axis bisector. Camera
translations interpolate linearly and rotations use spherical interpolation.
The source lenses stay fixed during the extrinsics sweep. Supply measured
side-camera transforms with `--source-sides`. When omitted, the side cameras
use the measured Frame side poses at both ends of the sweep.

Render each generated JSON with `handspotter-render --rig-json` in place
of `--calibration`. Use the same seed, source shards, scene directory, and
render options at every step. The output contains `handspotter.npy` in the order
side left, front left, front right, and side right. Each row is a full
four-view 128×160 byte tensor. The sweep scorer reports both per-view
coverage and results for the hand/camera pairs visible at every step.
Run `handspotter_camera_sweep.py score --datasets datasets --model handspotter-frame-dequantized.onnx --out benchmark.json`
with the rendered datasets named after their rig JSON files. Add
`--blank-side-views` to measure the front pair with zero-filled side slots.

### Staged Handspotter fine-tuning

Generate rigs using `handspotter_camera_sweep.py rigs --steps 0 2.5 5 7.5 10 25`
with the calibration arguments above. Render each training and validation
split at 0%, 2.5%, 5% and 7.5%. Render the test split at all six geometries.
Use disjoint pose assets and seeds between splits and reuse them across
geometries within each split. `--tensors-only` writes the model inputs and
annotations without also saving the rendered PNGs.

Generate a feature cache for each rendered dataset. Use `training`,
`validation`, or `test` as the filename prefix and split argument, and use
the rig filename stem for the geometry.

```sh
python train_handspotter.py cache --model handspotter-frame-integer.onnx \
  --synthetic synthetic/training-002.5 --split training \
  --out features/training-002.5.npz
python train_handspotter.py cache --model handspotter-frame-integer.onnx \
  --selection hot3d-split.json --clips hot3d-clips --split training \
  --out features/training-hot3d.npz
python train_handspotter.py train --model handspotter-frame-integer.onnx \
  --features features --out curriculum --evaluate
```

Each entry in the HOT3D selection names a `split`, `participant`, numeric
`clip`, archive `sha256` and list of `frames`. Assign whole participants
to one split. The loader resizes each calibrated grayscale front image to 160×128, scales native boxes into the detector window, and zero-fills the side slots. An optional `device` field checks the clip metadata. It also drops both
side images from a deterministic subset of synthetic training groups.

Training freezes the integer encoder through `conv10` and updates
`conv11`–`conv15` using straight-through gradients. Each round starts from
the previous selected checkpoint and mixes synthetic source-rig inputs, synthetic
shifted inputs, and real HOT3D inputs. It selects a checkpoint on all
three validation cohorts, including the incoming checkpoint as epoch zero.
The test set compares every selected checkpoint against the incoming model.
It reports all visible hands and the subset visible at every test geometry.
Use `--direct` in a separate output directory to spend the same training
budget at 0% and 7.5% throughout.
Each integer export runs through ONNX Runtime and is checked byte-for-byte on
validation inputs from both domains. Add `--decoded-model` to also produce
exports with the dequantized output interface.

For a direct endpoint experiment, use `--steps 100 --frame-lens` when
generating rigs. The `100` rig preserves the source lenses to isolate the
change in extrinsics. The `100-kb4` rig uses the calibrated Frame KB4 lenses
and maps each full sensor ROI to the Handspotter input window. Render and cache each
endpoint with its own suffix. Use `--direct-geometry 100` or
`--direct-geometry 100-kb4` to train at that endpoint in every round.
`--test-geometries 000 025 100 100-kb4` scores the same test poses across both
optics, with source and 25% geometry as references. HOT3D is included
automatically. Add `--reference LABEL PATH` for each earlier tail checkpoint
to benchmark on those same inputs.

Use `--steps 0 10 20 30 40 50 60 70 80 90 100 --morph-lenses` to change
the optics and the pose together. At each step the renderer blends the
source and resized Frame KB4 forward projections of the same ray.
The angular limits use the same fraction. The renderer numerically inverts
each intermediate profile and checks the pixel round trip. The endpoints
reuse the source 0% and calibrated `100-kb4` rigs.

Pass the nine `NNN-morphed` cache suffixes in order, followed by
`100-kb4`, to `train_handspotter.py train --schedule`. The trainer carries the
validation-selected weights from one stage into the next. Use
`--start-round` and `--rounds` to train through a growing set of feature
caches, and use the same number of rounds with
`--direct-geometry 100-kb4` for the control.

### Frame-only Handspotter training from scratch

Use the [packaged Handspotter workflow](../../docs/handspotter-training.md) to
render procedural hands or poses from a detailed mesh bank, mix the renders
with reviewed Frame footage, and train and export integer ONNX. The model's
architecture and export are defined in source. Use disjoint pose IDs and
recording sessions across splits.

### Warped HOT3D front-camera training

Use the public hand-labeled HOT3D-Clips archives from `train_quest3` and
`train_aria`. Both packages include the pinned hand-tracking camera toolkit.

```sh
nix run .#hot3d-frame -- --archive clip-000001.tar --rig frame-handspotter-rig.json --out warped-clip
nix run .#hot3d-frame-corpus -- --root corpus --rig frame-handspotter-rig.json
```

The corpus root must contain a `source.json` listing the Hugging Face repository, its immutable revision,
and each archive's path, byte size, device, and LFS SHA-256. The corpus
builder verifies archives and writes a training manifest after all clips
are ready. It can resume from existing output shards.

The warp aligns the source and Frame front-pair baselines and forward
bisectors. It maps each destination KB4 ray through the corresponding
HOT3D camera's FISHEYE624 projection and keeps the source optical center.
The sampled common field of view drives both the image and the warped
amodal boxes. Regions outside common coverage are black. The two upper
Frame slots are zero-filled and excluded from direct supervision.

Pass the finished corpus directory with `--hot3d-training` alongside
the synthetic `--training` shards. The trainer samples the union without
dropping groups, loads real JPEG-derived tensors by shard, and selects
checkpoints with the supplied Frame validation data.

### SOMA-X hands and wider reach

Install `py-soma-x` in the Python environment used for pose sampling and
provide the native SOMA-X asset directory. No MANO model file is required.
`sample_soma_hand.py --assets SOMA-assets --out soma-poses.npz --seed 48197`
draws hand identities and wrist-independent finger poses. Add
`--correctives SOMA-assets/correctives_model.pt` to apply the supplied pose
correctives. Use separate seeds for training and holdout banks.

Supply the resulting bank as both `--source soma-poses.npz` and
`--topology soma-poses.npz` to `handspotter-render`. The bank contains
the native mid-resolution right-hand triangles and 21 landmarks in the
renderer’s existing order. The renderer mirrors the mesh for the left hand,
and lofts the procedural forearm from the native wrist boundary.
`--wide-reach` works with either hand model and samples closer, higher, and
more lateral wrist placements within the arm length and elbow limits.
For a matched fine-tune, pass the same float `--initial-checkpoint` to
`handspotter-train` for each dataset.

### Articulated humanoids and fitted clothes

Both bakers use a rendered synthetic `dataset.json` as a scene template.
They place the eye joints behind the calibrated camera pair and use the
original wrist positions and elbow directions as reach targets. Each rig uses
its own bone lengths and a 43° wrist-bend limit. The manifest records wrist
position error in millimetres for each requested hand.

Use the body assets and pose correctives from SOMA-X with `py-soma-x` installed:

```sh
python sample_soma_body.py --assets SOMA-assets --correctives correctives_model.pt \
  --source template/dataset.json --out soma-body --count 96 --seed 681721
```

The body sampler defaults to stock SOMA-X skinning. The earlier
`--arm-rig anatomical` experiment is available for replay. It fits an oblique
radius–ulna rotation axis and radiocarpal and midcarpal pivots to each body's
neutral proportions. It soft-limits excessive roll before applying the SOMA-X
pose correctives and replaces the forearm and wrist skinning with local
dual-quaternion blending. The stock corrective is tapered around the carpal
rows and retained at the elbow and fingers. The exported hand and all 21
landmarks follow the distal pivot. The manifest records the requested and
effective forearm roll, the share of wrist rotation at the midcarpal pivot,
and wrist displacement from moving the pivot. The sampler checks the two
posed arm and hand surfaces for triangle intersections, tries small shoulder
rotations when they cross, and verifies clearance after re-skinning. The
manifest records the resulting shoulder adjustment for each arm.

Add `--cloth-reference` to export rest-pose vertices and garment-fitting joints
in the same camera-pair coordinates and vertex order as the posed skin mesh.
Pass `--soma-wardrobe balanced` with this body bank to cycle the nine styles,
or choose a style such as `wide-bell-sleeve`, `overlong-sweatshirt`, or
`wide-overlong-sleeve`. The overlong styles extend an open cuff over the
proximal hand. The solver settles the garment on the posed body once per group,
and the existing renderer uses that mesh for all cameras and hand masks.
Garment style, extension length, RMS vertex displacement from the fit, simulation timing and sampled skin clearance
are included in `dataset.json`. Older jobs requesting fitted garments automatically use draping. Draping requires WebGPU in the headless Chromium
provided by the `anyhand-pbr` shell.

### GarmentCode sewing patterns

The compiler uses the pinned GarmentCode panel mesher. It keeps the welded
simulation vertices at sewn edges, panel-space UVs at render corners, and
fabric rest lengths independently of the posed starting mesh. The runtime
fits the compiled torso and sleeves to each SOMA-X pose. The available
presets use the supplied GarmentCode body measurements. A pattern imported
from GarmentCodeData should use the measurement YAML shipped with that
sample. Pass `--design` for a custom GarmentCode design YAML.
The importer accepts the normalized GarmentCodeData specifications distributed
by AIpparel. When a pattern contains a separate lower garment, the importer
keeps the top. Pass `--upper-body-only` to keep the top when a lower garment
is sewn to it.

```sh
nix run "path:.#anyhand-garmentcode" -- --preset overlong --out wardrobe/long-sleeves.json
nix run "path:.#anyhand-garmentcode" -- \
  --spec sample_specification.json --body sample_body_measurements.yaml \
  --out wardrobe/dataset-top.json
```

Use `--garment-library wardrobe --soma-wardrobe dataset-top` for batch
rendering, or `--soma-wardrobe garmentcode-balanced` to cycle the compiled
library. For inspection in Studio, set `"garments": "/path/to/wardrobe"`
in the project file. The fitter is designed for tops and sleeves.
GarmentCodeData files retain their CC BY 4.0 attribution.
[GarmentCode](https://github.com/maria-korosteleva/GarmentCode) ·
[GarmentCodeData v2](https://doi.org/10.3929/ethz-b-000690432)

The GPU skin collider refits a BVH for each posed skin mesh and traverses
only triangles near each predicted cloth vertex. Timing separates collider
preparation (`collider_ms`) from draping and readback (`solve_ms`).
Clearance samples measure skin penetration at vertices and face centroids.
Compiled patterns use WebGPU distance constraints measured against the fabric's
rest lengths. The solver alternates these constraints with skin contact and
uses the nearby skin plane while projecting each edge. This reduces stretch
while letting the cloth fold. Procedural garments use the edge-spring solver.

### MPFB clothing

For the clothed variant, install the [MPFB extension](https://extensions.blender.org/add-ons/mpfb/)
in Blender's `user_default` extension repository and download the desired MHCLO
assets. The trial uses MPFB 2.0.18 with the CC0 [Shirts 01 pack](https://static.makehumancommunity.org/assets/assetpacks/shirts01.html).
`--garment` can be repeated. MPFB fits the rest meshes and transfers skinning
weights to each garment. PBR rendering applies the existing fabric material.

```sh
blender -b --python-exit-code 1 --python bake_mpfb_body.py -- \
  --source template/dataset.json --out mpfb-body --count 96 --seed 684483 \
  --garment path/to/short_sleeve.mhclo --garment path/to/long_sleeve.mhclo
```

Run `handspotter-render` in the `anyhand-pbr` shell with `--body-bank soma-body`
or `--body-bank mpfb-body`, the template's rig, Foundation assets, render seed and
group count. Pass `--capture-noise` to reproduce the template's lens effects.
Keep the render scale fixed across benchmark cohorts. The renderer derives
visible hand labels from the posed geometry and garment occlusion.

## Paired HOT3D replay

`replay_hot3d.py` uses one HOT3D clip to evaluate the appearance of
rendered AnyHand meshes. It retargets the meshes to the annotated UmeTrack
joints and projects both hands into the two calibrated views from the same
3D pose. Development frames select the source meshes and fit the grayscale
response. The replay fills the padded hand regions using aligned
donor frames from the clip, with inpainting as a fallback. Donor hand
regions are excluded before interpolation. With `--object-occlusion`, the
replay preserves the interiors of annotated visible objects and masks
those pixels out of the rendered hands. With `--harmonize`, it fits a
quantile tone curve and feathers plate seams outside the padded hand boxes.

Supply `--object-models` with the HOT3D
[object_models_eval](https://huggingface.co/datasets/bop-benchmark/hot3d/tree/main/object_models_eval)
GLB directory and install `trimesh` to use depth-ordered hand and object
compositing. This option includes the visible-object matte behavior.
Inside the erased area, missing object pixels can be recovered from the
other camera when the calibrated projection lands on the same object,
agrees in depth, and fits inside its visible mask. Objects without
verified depth and material support fall back to the visible-object matte.

Use `--pose-bank` to align the released AnyHand meshes to the recorded
joints and blend four nearby articulations in the shared MANO topology.
Both cameras use the resulting mesh and the same skin material.
`--forearms` joins short tapered skin meshes to the open wrist boundaries
and fades them into the existing forearms outside the removal region. The
hand masks exclude the added forearm geometry. The optional
`--surface-light` mode fits world-space lighting from development pixels
and adds mesh-space skin variation.

```sh
python replay_hot3d.py --clip clip-000000.tar \
  --calibration 000000.cameras.json --toolkit hand_tracking_toolkit_repo \
  --topology base_hand_right.obj --library /tmp/anyhand-raster.so \
  --source source-a.tar source-b.tar --fit 0 4 8 12 16 \
  --plate-donors 0 4 8 12 16 18 37 73 113 149 \
  --test $(seq 20 4 144) --object-models object_models_eval --harmonize \
  --forearms --pose-bank --out hot3d-replay
```

The output contains the capture, hand-erased plate, UmeTrack reference mesh
render, and AnyHand replay of each view, along with masks and the HOT3D
amodal boxes for each anatomical slot. Photographed scenery and any forearm pixels outside the removal
mask carry through to the replay. Use these pairs for renderer calibration
and use other clips and independent synthetic scenes to measure transfer.
