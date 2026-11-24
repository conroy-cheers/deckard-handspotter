{
  formats,
  runCommand,
  anyhand-soma-assets,
  anyhand-garment-library,
  anyhand-foundation-scenes,
  anyhand-starter-poses,
  handspotter-reference-calibration,
}:
let
  calibration = "${handspotter-reference-calibration}/share/handspotter/calibration";
  project = (formats.json { }).generate "render-job.json" {
    native_soma = true;
    randomize_poses = true;
    appearance_variation = true;
    capture_noise = true;
    slight_blur = true;
    lens_decentering = true;
    arm_motion_blur = true;
    camera_motion_blur = true;
    head_look = true;
    wide_reach = true;
    soma_hand_poses = "mixed";
    environment = "random";
    count = 1;
    soma_wardrobe = "random";
    cloth_mode = "drape";
    soma_assets = toString anyhand-soma-assets;
    soma_correctives = "${anyhand-soma-assets}/correctives_model.pt";
    soma_template = "${anyhand-starter-poses}/share/anyhand/templates/starter.json";
    rig_json = "${calibration}/rig.json";
    openvr_calibration = "${calibration}/openvr.json";
    garment_library = toString anyhand-garment-library;
    foundation_scenes = "${anyhand-foundation-scenes}/share/anyhand/foundation-scenes";
  };
in
runCommand "anyhand-render-project"
  {
    meta.description = "Default calibrated AnyHand render project with packaged inputs";
  }
  ''
    install -Dm444 ${project} "$out/share/anyhand/render-job.json"
  ''
