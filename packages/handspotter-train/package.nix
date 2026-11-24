{ mkTool, libraries }:
mkTool {
  pname = "handspotter-train";
  description = "Train and quantize a Handspotter hand detector";
  src = ./src;
  entryPoint = "train_handspotter_scratch";
  dependencies = [
    libraries.torchModel
    libraries.evaluation
  ];
}
