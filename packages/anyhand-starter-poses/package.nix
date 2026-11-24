{
  lib,
  formats,
  runCommand,
}:
let
  dataset = (formats.json { }).generate "starter.json" (import ./poses.nix { inherit lib; });
in
runCommand "anyhand-starter-poses"
  {
    meta.description = "Authored SOMA-X reach targets for AnyHand";
  }
  ''
    install -Dm444 ${dataset} "$out/share/anyhand/templates/starter.json"
  ''
