{
  mkTool,
  libraries,
  python3Packages,
  native,
  handspotter-render,
}:
mkTool {
  pname = "handspotter-prepare";
  description = "Prepare Handspotter dataset shards from recordings and renders";
  src = ./src;
  entryPoint = "handspotter_dataset";
  dependencies = [
    libraries.rig
    python3Packages.pillow
  ];
  runtimeInputs = [ handspotter-render ];
  environment.FRAME_HAND_PIPELINE_LIBRARY = "${native}/lib/libhandspotter-prepare.so";
}
