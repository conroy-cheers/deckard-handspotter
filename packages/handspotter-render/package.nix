{
  mkTool,
  lib,
  anyhand-soma-worker,
  anyhand-render-project,
  libraries,
  python3Packages,
  nodejs_24,
  chromium,
  pbr,
}:
mkTool {
  pname = "handspotter-render";
  description = "Render synthetic hands for calibrated Handspotter camera rigs";
  src = ./src;
  entryPoint = "render_handspotter";
  dependencies = [
    libraries.renderSupport
    python3Packages.pillow
    python3Packages.pycocotools
  ];
  tests = ./tests;
  extraInstall = ''
    ln -s ${../../lib/cloth}/*.mjs "$dir/"
  '';
  runtimeInputs = [
    nodejs_24
    chromium
  ];
  environment = {
    ANYHAND_RENDER_DEFAULTS = "${anyhand-render-project}/share/anyhand/render-job.json";
    ANYHAND_SOMA_WORKER = lib.getExe anyhand-soma-worker;
    inherit (pbr)
      ANYHAND_THREE_JS
      ANYHAND_CLOTH_JS
      ANYHAND_PBR_ASSETS
      ANYHAND_SKIN_TOPOLOGY
      ANYHAND_SKIN_KERNEL
      ;
  };
}
