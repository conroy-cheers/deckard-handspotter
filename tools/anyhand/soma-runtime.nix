{ pkgs }:
let
  p = pkgs.python313Packages;
  wheel =
    {
      pname,
      version,
      url,
      hash,
      name ? baseNameOf url,
      dependencies ? [ ],
      native ? false,
    }:
    p.buildPythonPackage {
      inherit pname version dependencies;
      format = "wheel";
      src = pkgs.fetchurl { inherit name url; sha256 = hash; };
      nativeBuildInputs = pkgs.lib.optional native pkgs.autoPatchelfHook;
      buildInputs = pkgs.lib.optionals native [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ];
      preFixup = pkgs.lib.optionalString native ''
        addAutoPatchelfSearchPath "$out/${p.python.sitePackages}"
      '';
    };
  torch = wheel {
    pname = "torch";
    version = "2.9.0+cpu";
    name = "torch-2.9.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl";
    url = "https://download-r2.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl";
    hash = "6c9b217584400963d5b4daddb3711ec7a3778eab211e18654fba076cce3b8682";
    native = true;
    dependencies = with p; [
      filelock
      fsspec
      jinja2
      networkx
      setuptools
      sympy
      typing-extensions
    ];
  };
  usd = wheel {
    pname = "usd-core";
    version = "26.11";
    url = "https://files.pythonhosted.org/packages/49/d0/aade15c70521eb0e87b9e8b0ffbe2024633b37f7243d1b98482fc9fb6ec6/usd_core-26.11-cp313-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "da42ec136033a69975655b44c65c75c6a494ddc8f1e6a52d57cb7cd2dc4603a8";
    native = true;
  };
  warp = wheel {
    pname = "warp-lang";
    version = "1.19.0";
    url = "https://files.pythonhosted.org/packages/17/e9/7da1fc1cd4b677477cc0d232351a9567dd69f45709da5600a91961e36eeb/warp_lang-1.19.0-py3-none-manylinux_2_28_x86_64.whl";
    hash = "c797f5f4aa9925f676e20dd28e5416fee794499f3f3c0a5b66f2d4dfc2de9e32";
    native = true;
    dependencies = [ p.numpy ];
  };
  cholespy = wheel {
    pname = "cholespy";
    version = "2.2.0";
    url = "https://files.pythonhosted.org/packages/84/94/b2a491bbd74386b2301ff8c3a33957261404f866846645e699b7611c65d6/cholespy-2.2.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "3e32e9760380a3804ac6da150ac4eb7e5b4554cfd3b36d437fd819c17e1e9317";
    native = true;
    dependencies = [ p.numpy ];
  };
  soma = wheel {
    pname = "py-soma-x";
    version = "0.3.6";
    url = "https://files.pythonhosted.org/packages/fe/92/2f88b4348c80a93f590c133650adcf0e3e4b8b20e5d353a8aed6e5edc827/py_soma_x-0.3.6-py3-none-any.whl";
    hash = "401ca64514f2dee843032bfdd4e94ee9ca59bf822568a486b320db209c22b147";
    dependencies = with p; [
      numpy
      scipy
      torch
      trimesh
      warp
      rtree
      cholespy
      huggingface-hub
      usd
    ];
  };
  python = pkgs.python313.withPackages (_: [ soma ]);
  sourceFiles = [
    "soma_worker.py"
    "soma_live.py"
    "soma_protocol.py"
    "soma_arm_collision.py"
    "soma_wardrobe.py"
    "garmentcode_fit.py"
    "sample_soma_hand.py"
    "humanoid_geometry.py"
    "head_pose.py"
    "replay_surface.py"
  ];
  workerSources = pkgs.runCommand "anyhand-soma-worker-sources" { } ''
    mkdir -p "$out"
    ${pkgs.lib.concatMapStringsSep "\n" (name: ''
      install -m644 ${./. + "/${name}"} "$out/${name}"
    '') sourceFiles}
  '';
in
pkgs.writeShellApplication {
  name = "anyhand-soma-worker";
  text = ''
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
    export OPENBLAS_NUM_THREADS=1
    exec ${python}/bin/python ${workerSources}/soma_worker.py "$@"
  '';
}
