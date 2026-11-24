{
  lib,
  python3Packages,
  cameraToolkit,
}:
let
  p = python3Packages;
  # Shared modules are installed once per Python package set. Accelerator variants
  # resolve Torch and its dependants in their own package set.
  library =
    pname: src: dependencies:
    p.buildPythonPackage {
      inherit pname src dependencies;
      version = "0.1.0";
      pyproject = false;
      dontBuild = true;
      installPhase = ''
        runHook preInstall
        install -Dt "$out/${p.python.sitePackages}" -m644 *.py
        runHook postInstall
      '';
    };
  pipeline = library "frame-camera-pipeline" (lib.fileset.toSource {
    root = ../pipeline;
    fileset = lib.fileset.fileFilter (f: f.hasExt "py") ../pipeline;
  }) [ p.numpy ];
  data = library "handspotter-data" ./data [ p.numpy ];
  model = library "handspotter-model" ./model [ p.onnx ];
  cameras = library "handspotter-cameras" ./cameras [
    data
    pipeline
    cameraToolkit
    p.scipy
    p.opencv4
  ];
  anyhand = ../tools/anyhand;
in
{
  inherit
    data
    model
    pipeline
    cameras
    ;
  rig = library "handspotter-rig-export" ./rig [
    data
    pipeline
  ];
  evaluation = library "handspotter-metrics" ./evaluation [
    data
    p.onnxruntime
  ];
  torchModel = library "handspotter-torch" ./torch [
    model
    p.torch
  ];
  hot3d = library "hot3d-warp" ./hot3d [ cameras ];
  renderSupport =
    library "anyhand-render-support"
      (lib.fileset.toSource {
        root = anyhand;
        fileset = lib.fileset.unions (
          map (name: anyhand + "/${name}.py") [
            "foundation_scenes"
            "garmentcode_fit"
            "head_pose"
            "humanoid_geometry"
            "render"
            "replay_surface"
            "soma_protocol"
            "soma_wardrobe"
          ]
        );
      })
      [
        cameras
        p.trimesh
        p.rtree
      ];
}
