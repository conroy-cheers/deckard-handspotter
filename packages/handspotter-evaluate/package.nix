{ mkTool, libraries }:
mkTool {
  pname = "handspotter-evaluate";
  description = "Evaluate a Handspotter ONNX model on prepared shards";
  src = ./src;
  entryPoint = "handspotter_evaluate";
  dependencies = [
    libraries.evaluation
    libraries.model
  ];
}
