{ mkTool, libraries }:
mkTool {
  pname = "hot3d-frame";
  description = "Warp a HOT3D clip into the two front Frame camera views";
  src = ./src;
  entryPoint = "hot3d_frame";
  dependencies = [ libraries.hot3d ];
  tests = ./tests;
}
