{
  mkTool,
  libraries,
  curl,
}:
mkTool {
  pname = "hot3d-frame-corpus";
  description = "Fetch and convert a pinned HOT3D corpus to Frame training shards";
  src = ./src;
  entryPoint = "hot3d_frame_corpus";
  dependencies = [ libraries.hot3d ];
  runtimeInputs = [ curl ];
}
