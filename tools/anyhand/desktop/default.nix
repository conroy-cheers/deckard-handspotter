{
  pkgs,
  projectFile ? null,
  workflow ? pkgs.callPackage ../../../nix/workflow.nix { inherit pbr; },
  renderer ? workflow.handspotter-render,
  pbr ? import ../pbr-shell.nix { inherit pkgs; },
}:
let
  inherit (pkgs) lib;
  worker = workflow.anyhand-soma-worker;
  somaAssets = workflow.anyhand-soma-assets;
  batch = lib.getExe renderer;
  starterProject = "${workflow.anyhand-render-project}/share/anyhand/render-job.json";
  studio = pkgs.runCommand "anyhand-studio-files" { } ''
    bundle="$out/libexec/anyhand-studio"
    mkdir -p "$bundle/frontend"
    install -m644 ${./main.cjs} "$bundle/main.cjs"
    install -m644 ${./preload.cjs} "$bundle/preload.cjs"
    install -m644 ${./batch.cjs} "$bundle/batch.cjs"
    ln -s ${batch} "$bundle/batch"
    ln -s ${
      pkgs.python3.withPackages (p: [
        p.numpy
        p.pillow
      ])
    }/bin/python3 "$bundle/preview-python"
    install -m644 ${../../../packages/handspotter-render/src/render_preview.py} "$bundle/preview.py"
    install -m644 ${./icons/icon.png} "$bundle/icon.png"
    cp ${./frontend}/* "$bundle/frontend/"
    ln -s ${starterProject} "$bundle/starter-project.json"
  '';
in
pkgs.symlinkJoin {
  name = "anyhand-studio";
  paths = [ studio ];
  nativeBuildInputs = [ pkgs.makeWrapper ];
  postBuild = ''
    mkdir -p "$out/bin"
    makeWrapper ${lib.getExe pkgs.electron} "$out/bin/.studio-gui" \
      --add-flags ${studio}/libexec/anyhand-studio/main.cjs \
      --set-default ANYHAND_GARMENT_LIBRARY ${pbr.ANYHAND_GARMENT_LIBRARY} \
      --prefix LD_LIBRARY_PATH : "${
        lib.makeLibraryPath [ pkgs.vulkan-loader ]
      }:/run/opengl-driver/lib" \
      ${lib.escapeShellArgs (
        lib.optionals (projectFile != null) [
          "--set-default"
          "ANYHAND_STUDIO_PROJECT"
          (toString projectFile)
        ]
      )}
    cat > "$out/bin/anyhand-studio" <<'SH'
    #!${pkgs.runtimeShell}
    case "''${1-}" in
      batch) shift; exec ${batch} "$@" ;;
      -h|--help)
        echo 'anyhand-studio                         Open the rendering workspace'
        echo 'anyhand-studio batch [RENDER OPTIONS]   Render without opening a window'
        echo 'anyhand-studio batch --job JOB --out DIR'
        exit 0 ;;
      "") ;;
      *) echo "Unknown Studio command: $1 (see --help)" >&2; exit 2 ;;
    esac
    exec "@gui@" "$@"
    SH
    substituteInPlace "$out/bin/anyhand-studio" --replace-fail @gui@ "$out/bin/.studio-gui"
    chmod +x "$out/bin/anyhand-studio"
  '';
  passthru = { inherit worker somaAssets; };
  meta.mainProgram = "anyhand-studio";
}
