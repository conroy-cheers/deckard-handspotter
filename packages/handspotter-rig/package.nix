{ mkTool, libraries }:
mkTool {
  pname = "handspotter-rig";
  description = "Export a four-camera Frame rig for Handspotter";
  src = ./src;
  entryPoint = "handspotter_rig";
  dependencies = [ libraries.rig ];
}
