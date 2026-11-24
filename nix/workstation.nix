{ inputs, lib, ... }:
{
  perSystem =
    {
      system,
      pkgs,
      config,
      ...
    }:
    let
      anyhand = ../tools/anyhand;
      pbr = import (anyhand + "/pbr-shell.nix") { inherit pkgs; };
      studio = pkgs.callPackage (anyhand + "/desktop") {
        inherit pbr workflow;
      };
      workflow = pkgs.callPackage ./workflow.nix { inherit pbr; };

      gpuPackages =
        nixpkgsConfig:
        let
          gpuPkgs = import inputs.nixpkgs {
            inherit system;
            config = nixpkgsConfig;
          };
          gpuWorkflow = gpuPkgs.callPackage ./workflow.nix { inherit pbr; };
          trainer = gpuWorkflow.handspotter-train;
        in
        {
          handspotter = workflow.handspotter.override { handspotter-train = trainer; };
          handspotter-train = trainer;
          frame-hand-labeler = gpuPkgs.callPackage ../tools/labeler { inherit (gpuWorkflow) libraries; };
        };
      accelerators = {
        rocm = gpuPackages { rocmSupport = true; };
        cuda = gpuPackages {
          cudaSupport = true;
          allowUnfree = true;
        };
      };
      gpuAliases = lib.concatMapAttrs (
        backend: lib.mapAttrs' (name: package: lib.nameValuePair "${name}-${backend}" package)
      ) accelerators;
    in
    lib.mkIf (system == "x86_64-linux") {
      legacyPackages = accelerators // {
        cpu = { inherit (config.packages) handspotter handspotter-train frame-hand-labeler; };
      };

      packages = gpuAliases // {
        inherit (workflow)
          handspotter
          handspotter-rig
          handspotter-reference-calibration
          anyhand-foundation-scenes
          anyhand-starter-poses
          anyhand-render-project
          anyhand-garment-library
          anyhand-soma-assets
          anyhand-soma-worker
          handspotter-prepare
          handspotter-render
          handspotter-train
          handspotter-export
          handspotter-evaluate
          hot3d-frame
          hot3d-frame-corpus
          ;
        default = workflow.handspotter;
        anyhand-studio = studio;
        anyhand-garmentcode = import (anyhand + "/garmentcode.nix") { inherit pkgs; };
      };

      checks = {
        inherit (config.packages)
          frame-hand-labeler-tools
          hot3d-frame
          hot3d-frame-corpus
          ;
        handspotter-workflow = workflow.handspotter.tests.smoke;
        anyhand-packaged-render =
          pkgs.runCommand "anyhand-packaged-render"
            {
              nativeBuildInputs = [
                studio
                pkgs.jq
              ];
            }
            ''
              export XDG_CACHE_HOME="$TMPDIR/cache" XDG_CONFIG_HOME="$TMPDIR/config" ANYHAND_SOFTWARE_RENDER=1
              mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
              anyhand-studio batch --seed 17 --count 2 --out sample --environment foundation-classroom --face-size 96
              jq -e '.records[0].environment.name == "foundation-classroom" and .records[0].wardrobe.mode == "drape" and .records[0].wardrobe.fit.solve_ms > 0 and .records[0].wardrobe.fit.rms_displacement_mm > 0 and (.stream_order | length == 4)' sample/dataset.json
              test -s sample/handspotter.npy
              ${pkgs.python3.withPackages (p: [ p.numpy p.pillow p.pycocotools ])}/bin/python3 ${../packages/handspotter-render/tests/viewport_session.py} sample/dataset.json
              touch "$out"
            '';
      };

      devShells = {
        anyhand-pbr = pbr.overrideAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ [ workflow.handspotter-render ];
        });
        default = pkgs.mkShell {
          packages = [
            workflow.handspotter-export
            workflow.handspotter-evaluate
            (pkgs.python3.withPackages (p: [
              workflow.libraries.renderSupport
              workflow.libraries.rig
              workflow.libraries.evaluation
              workflow.libraries.torchModel
              workflow.libraries.hot3d
              p.av
              p.numpy
              p.onnx
              p.onnxruntime
              p.opencv4
              p.pillow
              p.rtree
              p.scipy
              p.torch
              p.trimesh
            ]))
          ];
          PYTHONPATH = "${../pipeline}:${anyhand}";
        };
      };
    };
}
