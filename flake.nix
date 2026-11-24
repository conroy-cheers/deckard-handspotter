{
  description = "Handspotter labeling, training and export tools for the Steam Frame hand detector";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      imports = [ ./nix/workstation.nix ];

      perSystem = { pkgs, ... }:
      let
        libraries = pkgs.callPackage ./lib {
          cameraToolkit = pkgs.callPackage ./nix/hand-tracking-camera.nix { };
        };
      in {
        packages = {
          frame-hand-labeler-tools = pkgs.callPackage ./tools/labeler { inherit libraries; withModel = false; };
          frame-hand-labeler = pkgs.callPackage ./tools/labeler { inherit libraries; };
          frame-hand-labeler-desktop = pkgs.callPackage ./tools/labeler {
            inherit libraries;
            withModel = false;
            withDesktop = true;
          };
        };
        formatter = pkgs.nixfmt;
      };
    };
}
