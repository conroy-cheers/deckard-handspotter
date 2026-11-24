{
  lib,
  python3Packages,
  fetchFromGitHub,
}:
python3Packages.buildPythonPackage {
  pname = "hand-tracking-toolkit-camera";
  version = "0-unstable-2026-11-17";
  pyproject = false;
  src = fetchFromGitHub {
    owner = "facebookresearch";
    repo = "hand_tracking_toolkit";
    rev = "1aed9608616f62b6468c9c1a1042d60e9c210c70";
    hash = "sha256-dYS69GqZ+XscJ8QKjateWnJaksgN+XrzQI2sqwrmDak=";
  };
  dependencies = with python3Packages; [
    numpy
    scipy
    typing-extensions
  ];
  dontBuild = true;
  installPhase = ''
    runHook preInstall
    install -Dt "$out/${python3Packages.python.sitePackages}/hand_tracking_toolkit" \
      -m644 hand_tracking_toolkit/{__init__,camera,camera_distortion,math_utils}.py
    install -Dm644 LICENSE "$out/share/licenses/hand-tracking-toolkit-camera/LICENSE"
    runHook postInstall
  '';
  pythonImportsCheck = [ "hand_tracking_toolkit.camera" ];
  meta = {
    description = "Camera projections from Meta's hand tracking toolkit";
    homepage = "https://github.com/facebookresearch/hand_tracking_toolkit";
    license = lib.licenses.asl20;
  };
}
