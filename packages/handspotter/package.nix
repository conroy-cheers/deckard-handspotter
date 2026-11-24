{
  lib,
  writeShellApplication,
  runCommand,
  python3,
  libraries,
  handspotter-rig,
  handspotter-render,
  handspotter-prepare,
  handspotter-train,
  handspotter-export,
  handspotter-evaluate,
}:
let
  cli = writeShellApplication {
    name = "handspotter";
    runtimeInputs = [
      handspotter-rig
      handspotter-render
      handspotter-prepare
      handspotter-train
      handspotter-export
      handspotter-evaluate
    ];
    meta = {
      description = "Run the Handspotter training workflow";
      license = lib.licenses.mit;
      mainProgram = "handspotter";
    };
    text = ''
      case "${"$"}{1:---help}" in
        help|--help|-h)
          cat <<'HELP'
      Usage: handspotter COMMAND [options]

        rig       Export a calibrated camera rig
        render    Generate synthetic training images
        prepare   Assemble recorded and synthetic dataset shards
        train     Train and quantize a detector
        export    Convert a folded checkpoint to ONNX
        evaluate  Score an ONNX model against labeled data

      Examples:
        handspotter prepare --config dataset.toml --out dataset
        handspotter train --config train.toml --device cpu

      Run handspotter COMMAND --help for stage options.
      HELP
          ;;
        rig|render|prepare|train|export|evaluate) command="handspotter-$1"; shift; exec "$command" "$@" ;;
        *) echo "Unknown Handspotter command: $1. Run handspotter help for available commands." >&2; exit 2 ;;
      esac
    '';
  };
  python = python3.withPackages (p: [
    p.av
    libraries.data
    libraries.model
    libraries.pipeline
    p.numpy
    p.pillow
    p.onnx
    p.onnxruntime
  ]);
in
cli.overrideAttrs (old: {
  passthru = (old.passthru or { }) // {
    tests.smoke =
      runCommand "handspotter-cpu-workflow"
        {
          nativeBuildInputs = [
            cli
            handspotter-prepare
          ];
        }
        ''
          export ANYHAND_SOFTWARE_RENDER=1
          export XDG_CONFIG_HOME="$TMPDIR/chromium-config"
          export XDG_CACHE_HOME="$TMPDIR/chromium-cache"
          mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
          export PYTHONPATH=${../../tools/labeler}
          ${python}/bin/python -B ${./tests/test_workflow.py}
          touch "$out"
        '';
  };
})
