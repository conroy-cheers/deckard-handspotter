{ mkTool, libraries }:
mkTool {
  pname = "handspotter-export";
  description = "Export folded Handspotter checkpoints to ONNX";
  src = ./src;
  entryPoint = "handspotter_export";
  dependencies = [ libraries.torchModel ];
}
