{ pkgs }:
let
  download = url: hash: pkgs.fetchurl { inherit url hash; };
  threeSource = download "https://registry.npmjs.org/three/-/three-0.187.0.tgz" "sha512-Qiu7R+Cmh8CstZsY1uJOSWnVlfo7ZtNLV7mF8Jltk3LB3AtDu9cRZlcnRKkAk8Y4z6KKw35z7Y7Urd7jWY8etA==";
  three = pkgs.runCommand "anyhand-three-0.187.0" { } ''
    mkdir "$out"
    tar -xzf ${threeSource} --strip-components=1 -C "$out" \
      package/build package/examples/jsm/controls/OrbitControls.js package/examples/jsm/loaders/HDRLoader.js \
      package/examples/jsm/loaders/GLTFLoader.js \
      package/examples/jsm/lights/RectAreaLightUniformsLib.js \
      package/examples/jsm/lights/RectAreaLightTexturesLib.js \
      package/examples/jsm/utils/BufferGeometryUtils.js \
      package/examples/jsm/utils/SkeletonUtils.js package/LICENSE
  '';
  bvhSource = download "https://registry.npmjs.org/three-mesh-bvh/-/three-mesh-bvh-0.9.18.tgz" "sha256-sEG/mrWbkMphNPzz3DHoQ6rAhSs4A6t8og3AyKEIPNw=";
  cloth = pkgs.runCommand "anyhand-cloth-dependencies" { } ''
    mkdir -p "$out" bvh
    tar -xzf ${bvhSource} --strip-components=1 -C bvh package/build/index.module.js package/LICENSE
    install -m644 bvh/build/index.module.js "$out/mesh-bvh.mjs"
    install -m644 bvh/LICENSE "$out/three-mesh-bvh.LICENSE"
  '';
  skin = download "https://www.texturecan.com/downloads/skin_0001/skin_0001_1k_tGTv2h.zip" "sha256-UrSFQJsWxtdMD2pGnuE5JdaIcYueTqLeKTex3R3uhj0=";
  cdn = "https://dl.polyhaven.org/file/ph-assets";
  texture =
    name: channel: hash:
    download "${cdn}/Textures/jpg/1k/${name}/${name}_${channel}_1k.jpg" hash;
  hdr = name: hash: download "${cdn}/HDRIs/hdr/1k/${name}_1k.hdr" hash;
  maps = {
    "wood-color.jpg" =
      texture "wood_table_large" "diff"
        "sha256-0Gr9AGS9bzu47mpm/ULP0IXTNVL/wiW/QeEJ3NqPKQE=";
    "wood-normal.jpg" =
      texture "wood_table_large" "nor_gl"
        "sha256-ZESKpDRMvYf0S5r5KshukUmws5Ef7POIGoMPbAuUnY8=";
    "wood-roughness.jpg" =
      texture "wood_table_large" "rough"
        "sha256-8rCum/gm7Vn7i4J/XAHfk1u7Sh7zP1FnrYuHVlOGtVk=";
    "concrete-color.jpg" =
      texture "concrete_floor_02" "diff"
        "sha256-bbhKbYfcUFCLnRNNb70rfodrgIGl9DoRiiScoY3nOvU=";
    "concrete-normal.jpg" =
      texture "concrete_floor_02" "nor_gl"
        "sha256-/No5G1eoFtZGrhVURoi+U3kEeNpc1I2+UUs8aQFBXmE=";
    "concrete-roughness.jpg" =
      texture "concrete_floor_02" "rough"
        "sha256-x/AmphG3hZkDxh7H/UV0lEFXIj6ed5ZBPdbipm6urHo=";
    "fabric-color.jpg" =
      texture "fabric_pattern_07" "col_1"
        "sha256-rrehSHt3rsVeOd5mLe9NNx9cfGF/YtHykdf5ky/+vck=";
    "fabric-normal.jpg" =
      texture "fabric_pattern_07" "nor_gl"
        "sha256-9pVbverZ8CPyNX9viAjtSWJ9EIgs0qPLa1pazhVavEU=";
    "fabric-roughness.jpg" =
      texture "fabric_pattern_07" "rough"
        "sha256-PZCsQLXDKhVknMhSW37u0TPqRrY6Dj6uXjSRHVyWfjo=";
    "desk.hdr" = hdr "poly_haven_studio" "sha256-38hQV2EBjWRJl6gD8zIzkyivK1PoICP7O+w3zv30O4M=";
    "workbench.hdr" = hdr "small_workshop" "sha256-ZpPCePKQSj9FDuQxclxgXAXVdWcDB9oYfNgl4hwT5MA=";
  };
  materials = pkgs.runCommand "anyhand-pbr-materials" { nativeBuildInputs = [ pkgs.unzip ]; } ''
    mkdir "$out"
    unzip -p ${skin} skin_0001_color_1k.jpg > "$out/skin-color.jpg"
    unzip -p ${skin} skin_0001_roughness_1k.jpg > "$out/skin-roughness.jpg"
    unzip -p ${skin} skin_0001_height_1k.png > "$out/skin-height.png"
    ${pkgs.lib.concatStringsSep "\n" (
      pkgs.lib.mapAttrsToList (name: path: ''ln -s ${path} "$out/${name}"'') maps
    )}
  '';
in
pkgs.mkShell {
  packages = [
    pkgs.nodejs_24
    pkgs.chromium
    (pkgs.python3.withPackages (p: [
      p.numpy
      p.scipy
      p.opencv4
      p.trimesh
      p.rtree
      p.typing-extensions
    ]))
  ];
  ANYHAND_THREE_JS = three;
  ANYHAND_CLOTH_JS = cloth;
  ANYHAND_PBR_ASSETS = materials;
  ANYHAND_SKIN_TOPOLOGY = import ./skin-collider-topology.nix { inherit pkgs three cloth; };
  ANYHAND_SKIN_KERNEL = import ./skin-collider-kernel.nix { inherit pkgs; };
}
// pkgs.lib.optionalAttrs (pkgs.stdenv.hostPlatform.system == "x86_64-linux") {
  ANYHAND_GARMENT_LIBRARY = import ./garment-library.nix { inherit pkgs; };
}
