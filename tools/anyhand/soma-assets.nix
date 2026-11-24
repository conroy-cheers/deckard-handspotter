{ pkgs }:
let
  revision = "75acbe89a5082d2d36f1150bde8fb7d7b8d77f42";
  source =
    file: sha256:
    pkgs.fetchurl {
      url = "https://huggingface.co/nvidia/SOMA-X/resolve/${revision}/${file}";
      inherit sha256;
    };
  assets = {
    LICENSE = "d1a7d615ab8eff4de143b1456f46dabf232f54daf0fcf9a70442bb6f637a9e95";
    "SOMAHand.npz" = "233e9225c1da05c6a3dd98107b9f027f2366e3b972bbe67e0d164e5d28619fff";
    "SOMA_neutral.npz" = "5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63";
    "SOMA_procedural_transforms.json" =
      "9853e19ab51625ffaa7d9e7ccf0e443272c2740837604e6bacd66604e8edf509";
    "SOMA_template_rig.usda" = "a952cb7bdf4ccd801e355d53122d395b985b69d0d9181439edad5081bb171856";
    "correctives_model.pt" = "95d61f4b35cfc287b2288c67349aa2128b710680e8fea333b04c42c95b98af31";
  };
in
pkgs.runCommand "anyhand-soma-assets-${builtins.substring 0 7 revision}"
  {
    meta.license = pkgs.lib.licenses.asl20;
    meta.homepage = "https://huggingface.co/nvidia/SOMA-X";
  }
  ''
    mkdir -p "$out"
    ${pkgs.lib.concatMapAttrsStringSep "\n" (file: sha256: ''
      ln -s "${source file sha256}" "$out/${file}"
    '') assets}
  ''
