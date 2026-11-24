{
  runCommand,
  python3,
  libraries,
}:
runCommand "handspotter-reference-calibration"
  {
    nativeBuildInputs = [ (python3.withPackages (_: [ libraries.rig ])) ];
    meta.description = "Matched four-camera rig and OpenVR mount for reference Steam Frame rendering";
  }
  ''
    dest="$out/share/handspotter/calibration"
    mkdir -p "$dest"
    cd "$dest"
    python - <<'PY'
    import json
    from pathlib import Path

    from calibration import load_frame
    from rig_export import export_rig

    cameras = Path("${./cameras.json}")
    alignment = Path("${./alignment.json}")
    export_rig(cameras, alignment, Path("rig.json"))
    transforms = {name: {"parent_from_eye": camera.transform.tolist()}
                  for name, camera in load_frame(cameras, alignment).items()}
    Path("openvr.json").write_text(json.dumps({"cameras": transforms}, indent=2) + "\n")
    PY
  ''
