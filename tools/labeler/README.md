# Frame Hand Labeler

Label hands in a [recorded Frame take](../../docs/frame-takes.md). The workspace keeps native
grayscale pixels, four-camera group IDs and shutter times. Each camera is
labeled independently. Mark the subject’s anatomical left or right hand.

Launch the Electron desktop app with `nix run .#frame-hand-labeler`. Use
**Open workspace** for an existing workspace, or import a recording or HOT3D
clip from the start screen. **File → Export labels** writes grouped JSON and
COCO into the current workspace's `labels/` directory. Open a workspace
directly with `nix run .#frame-hand-labeler -- gui ./clip`.

The `frame-hand-labeler-desktop` package provides the desktop app without the
SAM model. The `frame-hand-labeler-tools` package provides the CLI and browser
editor without Electron. The same `ingest`, `serve`, and `export` commands
work in all three packages.

```sh
nix run .#frame-hand-labeler -- ingest "/path/to/take" ./clip --every 4
nix run .#frame-hand-labeler -- serve ./clip
```

The take directory must contain `tracking.nut` and `timestamps.csv`. Use
`--start SECONDS --duration SECONDS` to import an excerpt. The importer verifies
packet timestamps against the sidecar and keeps the cameras together when
sampling. All frame pixels are saved as native-orientation grayscale PNGs.

### Import a HOT3D clip

Pass a HOT3D-Clips archive to the same `ingest` command. By default it
imports tracking streams `1201-1` and `1201-2` as `slam_left` and
`slam_right`. Repeat `--channel-map SOURCE=FRAME_CAMERA` to select and name
channels explicitly. The workspace contains only the selected views.

```sh
nix run .#frame-hand-labeler-tools -- ingest clip-000000.tar ./hot3d-clip \
  --warp-rig frame-handspotter-rig.json
nix run .#frame-hand-labeler-tools -- serve ./hot3d-clip
nix run .#frame-hand-labeler-tools -- export ./hot3d-clip
```

Omit `--warp-rig` to keep the source camera geometry. With it, the labeler
uses the HOT3D training warp to remap the tracking views and boxes into
160×128 Frame front images. Both modes preserve synchronized groups and
source timestamps and seed editable labels from HOT3D. Imported amodal
boxes carry `box_mode: "amodal"` in both exports. Missing annotations and
uncertain hands stay unknown, and the COCO export omits those views.

Use `hot3d-frame-corpus` for the existing NumPy training shards. The
labeler's calibration-driven contact and handedness tools require a native
Frame recording.

### Review and export labels

For footage recorded on a Frame, build or copy the AArch64 `frame-hand-labeler-tools`
package and run its `frame-hand-labeler ingest` on the headset. Transfer the resulting
workspace to your workstation. This avoids transferring the original
uncompressed video and keeps the model off the headset. The tools package
also provides the editor and export commands.

Choose a camera and a hand, then draw a tight box around the visible hand.
Manual boxes seed SAM. Mark a hand **Absent** to stop propagation across that
frame, or **Unknown** if the view is ambiguous. Seed each camera separately and
add another seed when a hand reappears.

In **SAM 3.1 tracking**, set the first and last groups directly or use **Set in**
and **Set out** at the playhead. Choose the current camera or all cameras and
click **Run SAM 3.1**. The radius is measured in groups on either side of a seed.
The timeline marks seeds, proposals, and reviewed slots for the active camera
and hand. Use **Next proposal** to jump to a proposal and **Accept + next** to
save it and continue reviewing. A manual redraw replaces a proposal.

```sh
nix run .#frame-hand-labeler -- propagate ./clip --radius 30
```

The Nix package includes the fixed-hash **SAM 3.1 Multiplex** checkpoint and
ComfyUI's inference modules. It can run entirely offline after building. On an
AMD workstation, launch the desktop with `nix run .#frame-hand-labeler-rocm`
or add `-- propagate ./clip` for the CLI. Use `frame-hand-labeler-cuda` for
NVIDIA GPUs. New proposals record the checkpoint hash alongside the seed.
Use `--device cuda` to require the GPU and `--threads N` to bound CPU threads.
Use `--camera slam_left` to iterate on one view. The radius counts sampled frames on each side of a
manual seed. The tracker uses native camera coordinates and keeps proposals
separate from reviewed labels. An empty predicted mask leaves the hand
unknown.

Click **Reload labels** in the editor. Dashed boxes are proposals.
Accept a visible box, redraw it, or mark it absent or unknown. The editor
stores each change immediately. Unreviewed proposals stay out of exports.

```sh
nix run .#frame-hand-labeler -- export ./clip
```

`clip/labels/groups.json` preserves group membership, per-camera timestamps,
human-reviewed boxes and separate absent and unknown states. Boxes use
half-open native pixel edges `[x1,y1,x2,y2]`. The take’s camera luma and
orientation are preserved.

`clip/labels/coco.json` has one image per fully reviewed view, including
reviewed negative views. It omits views with an unknown hand because COCO
does not encode unknown labels per class. Both exports resolve image paths
relative to `clip/`. Keep entire recording sessions together when dividing
the data into training and validation sets.

## Terra review

Choose **Terra review**, set the clip range and review rate, then click **Run
Terra**. It uses your signed-in Codex CLI and sends all available camera views
for each sampled group to GPT-6.1 Terra. The labeler selects the nearest
available group at the requested cadence. Completed groups are reused on
subsequent runs. Use **Stop** to cancel a run and keep its completed requests.

```sh
nix run .#frame-hand-labeler-tools -- ingest "/path/to/take" ./clip --hz 10
nix run .#frame-hand-labeler-tools -- terra ./clip
nix run .#frame-hand-labeler-tools -- terra ./clip --start-index 30 --limit 60 --hz 5
nix run .#frame-hand-labeler-tools -- resolve-hands ./clip \
  --xrservice /path/to/xrservice.json --device-config /path/to/device_config.json
nix run .#frame-hand-labeler-tools -- review-contacts ./clip \
  --xrservice /path/to/xrservice.json --device-config /path/to/device_config.json
nix run .#frame-hand-labeler-tools -- serve ./clip
nix run .#frame-hand-labeler-tools -- export ./clip --include-terra
```

The editor shows Terra boxes and its absent or unsure verdicts. Human edits
take precedence. Use `--include-terra` to include Terra results in exports.
Grouped JSON preserves the model provenance and uncertain hands. COCO omits
views with an uncertain hand.

Use the recording headset's own calibration for `resolve-hands`. It matches
boxes across the native camera views and follows them across adjacent
timestamps. Head-relative 3D positions also bridge motion when boxes lose
overlap in the images, and help establish the orientation of a track.
Use `--anchors WORKSPACE` to supply
manually annotated views from the same recording, including excerpts sampled at
a different cadence. An anchored track retains its identity as hands cross.
The editor and exports pick up the corrected identities from
`terra-handedness.json` after a reload. Isolated or ambiguous tracks can still
require human review.

`review-contacts` flags overlapping boxes when the front stereo pair
cannot confidently confirm their labels. It groups nearby flags, adds two
frames of context per side, and sends both SLAM views to Terra. The review
traces each hand back through its wrist and forearm. Use `--plan-only` to
check the number of groups before making requests. The review resumes after
an interruption. A completed review overlays the model's earlier boxes in
the editor and exports, with human labels taking precedence. An unresolved
hand is recorded as unknown.
