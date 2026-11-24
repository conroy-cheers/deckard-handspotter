{
  lib,
  stdenv,
}:
let
  pipeline = ../pipeline;
in
stdenv.mkDerivation {
  pname = "handspotter-camera-preparation";
  version = "0.1.0";
  src = lib.fileset.toSource {
    root = pipeline;
    fileset = lib.fileset.unions [
      (pipeline + "/handspotter_prepare.cpp")
      (pipeline + "/handspotter_prepare.h")
    ];
  };
  dontConfigure = true;
  buildPhase = ''
    runHook preBuild
    $CXX -std=c++20 -O2 -shared -fPIC handspotter_prepare.cpp -o libhandspotter-prepare.so
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    install -Dm755 libhandspotter-prepare.so "$out/lib/libhandspotter-prepare.so"
    runHook postInstall
  '';
  meta = {
    description = "Native Handspotter camera preprocessing library";
    license = lib.licenses.mit;
    platforms = lib.platforms.linux;
  };
}
