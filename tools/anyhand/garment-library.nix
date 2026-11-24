{ pkgs }:
let
  compiler = import ./garmentcode.nix { inherit pkgs; };
in
pkgs.runCommand "anyhand-garment-library" { nativeBuildInputs = [ compiler ]; } ''
  mkdir -p "$out"
  export MPLCONFIGDIR="$TMPDIR/matplotlib"
  anyhand-garmentcode --preset tee --out "$out/garmentcode-tee.json"
  anyhand-garmentcode --preset overlong --out "$out/garmentcode-overlong.json"
''
