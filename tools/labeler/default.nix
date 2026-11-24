{
  lib,
  stdenvNoCC,
  python3,
  fetchurl,
  makeWrapper,
  electron,
  copyDesktopItems,
  makeDesktopItem,
  comfyui,
  libraries,
  withModel ? true,
  withDesktop ? withModel,
}:
let
  weights = fetchurl {
    url = "https://huggingface.co/Comfy-Org/sam3.1/resolve/7bb8374780a725b4353ed31f3a9395c9742b5621/checkpoints/sam3.1_multiplex_fp16.safetensors";
    sha256 = "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03";
  };
  python = python3.withPackages (
    p:
    [
      p.av
      p.numpy
      p.pillow
      libraries.hot3d
    ]
    ++ lib.optionals withModel [
      p.comfy-aimdo
      p.comfy-kitchen
      p.einops
      p.psutil
      p.requests
      p.safetensors
      p.scipy
      p.sentencepiece
      p.torch
      p.torchsde
      p.torchvision
      p.transformers
    ]
  );
in
stdenvNoCC.mkDerivation {
  pname =
    if withModel then "frame-hand-labeler"
    else if withDesktop then "frame-hand-labeler-desktop"
    else "frame-hand-labeler-tools";
  version = "0.1.0";
  src = lib.fileset.toSource {
    root = ../..;
    fileset = lib.fileset.unions [
      (lib.fileset.fileFilter (
        f: f.hasExt "py" || f.hasExt "html" || f.hasExt "css" || f.hasExt "js" || f.hasExt "cjs" || f.name == "README.md"
      ) ./.)
      ../../pipeline/calibration.py
      ../../pipeline/geometry.py
    ];
  };
  sourceRoot = "source/tools/labeler";
  nativeBuildInputs = [ makeWrapper ] ++ lib.optional withDesktop copyDesktopItems;
  desktopItems = lib.optional withDesktop (makeDesktopItem {
    name = "frame-hand-labeler";
    desktopName = "Frame Hand Labeler";
    exec = "frame-hand-labeler gui";
    categories = [ "Graphics" ];
  });
  dontBuild = true;
  doCheck = stdenvNoCC.buildPlatform.canExecute stdenvNoCC.hostPlatform;
  checkPhase = ''
    runHook preCheck
    PYTHONPATH=../../pipeline ${python}/bin/python3 -B -m unittest -v
    ${python}/bin/python3 -B app.py --help
    ${lib.optionalString withModel ''
      PYTHONPATH=${comfyui.src} ${python}/bin/python3 -B -c '
      from comfy.cli_args import args
      args.cpu = True
      from comfy.sd import load_diffusion_model
      from comfy.ldm.sam3.tracker import SAM31Tracker, unpack_masks'
    ''}
    runHook postCheck
  '';
  installPhase = ''
    runHook preInstall
    install -Dt "$out/share/frame-hand-labeler" -m644 app.py model_jobs.py tracker.py terra.py handedness.py contact.py workspace.py hot3d_input.py index.html editor.js editor.css README.md
    install -Dt "$out/share/frame-hand-labeler" -m644 ../../pipeline/calibration.py ../../pipeline/geometry.py
    cli="$out/bin/frame-hand-labeler"
    ${lib.optionalString withDesktop ''cli="$out/libexec/frame-hand-labeler-cli"''}
    makeWrapper ${python}/bin/python3 "$cli" \
      --add-flags "-B $out/share/frame-hand-labeler/app.py" \
      ${lib.optionalString withModel ''
        --prefix PYTHONPATH : ${comfyui.src} \
        --set-default FRAME_HAND_LABELER_WEIGHTS ${weights}
      ''}
    ${lib.optionalString withDesktop ''
      install -Dt "$out/share/frame-hand-labeler/desktop" -m644 desktop/*
      makeWrapper ${lib.getExe electron} "$out/libexec/frame-hand-labeler-gui" \
        --add-flags "$out/share/frame-hand-labeler/desktop/main.cjs" \
        --set FRAME_HAND_LABELER_CLI "$cli" \
        --prefix LD_LIBRARY_PATH : /run/opengl-driver/lib
      mkdir -p "$out/bin"
      cat > "$out/bin/frame-hand-labeler" <<'SH'
      #!${stdenvNoCC.shell}
      case "''${1-}" in
        "") exec "@out@/libexec/frame-hand-labeler-gui" ;;
        gui)
          shift
          if [ "''${1-}" = "--help" ] || [ "''${1-}" = "-h" ]; then
            echo 'Usage: frame-hand-labeler gui [WORKSPACE]'
            exit 0
          fi
          if [ "$#" -gt 1 ]; then
            echo 'Usage: frame-hand-labeler gui [WORKSPACE]' >&2
            exit 2
          fi
          exec "@out@/libexec/frame-hand-labeler-gui" "$@" ;;
        *) exec "@out@/libexec/frame-hand-labeler-cli" "$@" ;;
      esac
      SH
      substituteInPlace "$out/bin/frame-hand-labeler" --replace-fail @out@ "$out"
      chmod +x "$out/bin/frame-hand-labeler"
    ''}
    runHook postInstall
  '';
  meta = {
    description = "Review and export hand boxes from Steam Frame recordings and HOT3D clips";
    mainProgram = "frame-hand-labeler";
    platforms = lib.platforms.linux;
  };
}
