{
  lib,
  newScope,
  pkgs,
  pbr,
}:
lib.makeScope newScope (self: {
  inherit pbr;
  anyhand-soma-assets = import ../tools/anyhand/soma-assets.nix { inherit pkgs; };
  anyhand-soma-worker = import ../tools/anyhand/soma-runtime.nix { inherit pkgs; };
  anyhand-garment-library = pbr.ANYHAND_GARMENT_LIBRARY;
  anyhand-render-project = self.callPackage ../packages/anyhand-render-project/package.nix { };
  mkTool = self.callPackage ./python-tool.nix { };
  native = self.callPackage ./camera-preparation.nix { };
  cameraToolkit = self.callPackage ./hand-tracking-camera.nix { };
  libraries = self.callPackage ../lib { };
  handspotter-reference-calibration =
    self.callPackage ../packages/handspotter-reference-calibration/package.nix
      { };
  anyhand-foundation-scenes = self.callPackage ../packages/anyhand-foundation-scenes/package.nix { };
  anyhand-starter-poses = self.callPackage ../packages/anyhand-starter-poses/package.nix { };
  handspotter-rig = self.callPackage ../packages/handspotter-rig/package.nix { };
  handspotter-prepare = self.callPackage ../packages/handspotter-prepare/package.nix { };
  handspotter-render = self.callPackage ../packages/handspotter-render/package.nix { };
  handspotter-train = self.callPackage ../packages/handspotter-train/package.nix { };
  handspotter-export = self.callPackage ../packages/handspotter-export/package.nix { };
  handspotter-evaluate = self.callPackage ../packages/handspotter-evaluate/package.nix { };
  hot3d-frame = self.callPackage ../packages/hot3d-frame/package.nix { };
  hot3d-frame-corpus = self.callPackage ../packages/hot3d-frame-corpus/package.nix { };
  handspotter = self.callPackage ../packages/handspotter/package.nix { };
})
