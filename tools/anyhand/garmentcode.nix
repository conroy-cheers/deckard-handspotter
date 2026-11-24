{ pkgs }:
let
  python = pkgs.python312;
  p = python.pkgs;
  revision = "d449629979028123a5c4dc9e732a2ec19b7fce31";
  source = pkgs.fetchurl {
    url = "https://codeload.github.com/maria-korosteleva/GarmentCode/tar.gz/${revision}";
    name = "GarmentCode-${revision}.tar.gz";
    hash = "sha256-fwf5X5sXAELhOz0EaNqLVmKv5E5aZ6GhQb/jxUWoERc=";
  };
  wheel =
    pname: version: url: hash: dependencies:
    p.buildPythonPackage {
      inherit pname version dependencies;
      format = "wheel";
      src = pkgs.fetchurl { inherit url hash; };
      nativeBuildInputs = [ pkgs.autoPatchelfHook ];
      buildInputs = [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ];
      preFixup = ''addAutoPatchelfSearchPath "$out/${python.sitePackages}"'';
    };
  cgal =
    wheel "cgal" "6.0.1.post202410241521"
      "https://files.pythonhosted.org/packages/02/59/4f904e7fca4b2c90fa9692737faaf93f5c192b143a66c407d6b30ce57b13/cgal-6.0.1.post202410241521-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
      "sha256-CRTCVc0Ml/YzD3T2X2SyDwmQaCA8FHtXKDVZnxcYNBw="
      [ p.numpy ];
  igl =
    wheel "libigl" "2.6.3"
      "https://files.pythonhosted.org/packages/f6/c2/8641402402fbfa553c51615825434802de9c3b60b2580acdaad5312af396/libigl-2.6.3-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
      "sha256-Sh0orr/riyMqWJQ7GvMY7GGh//dlCp0loTlrqPoe9tY="
      [
        p.numpy
        p.scipy
      ];
  runtime = python.withPackages (
    ps: with ps; [
      numpy
      scipy
      pyyaml
      svgwrite
      svgpathtools
      psutil
      matplotlib
      cairosvg
      cgal
      igl
    ]
  );
  scripts = pkgs.runCommand "anyhand-garmentcode-sources" { } ''
    mkdir -p "$out"
    tar -xf ${source} -C "$out" --strip-components=1 \
      GarmentCode-${revision}/pygarment \
      GarmentCode-${revision}/assets/garment_programs \
      GarmentCode-${revision}/assets/design_params \
      GarmentCode-${revision}/assets/bodies/body_params.py \
      GarmentCode-${revision}/assets/bodies/mean_all.yaml \
      GarmentCode-${revision}/LICENSE
    # Replace scalar 2D cross products removed by NumPy 2. The 3D
    # cross products used to orient panels and normals are untouched.
    sed -i '0,/cross = np.cross(v1, v2)/s//cross = v1[0]*v2[1] - v1[1]*v2[0]/' "$out/pygarment/garmentcode/utils.py"
    substituteInPlace "$out/pygarment/pattern/utils.py" \
      --replace-fail 'np.cross(v1, v2)' 'v1[0]*v2[1] - v1[1]*v2[0]' \
      --replace-fail 'np.cross(edge_vec, point_vec)' 'edge_vec[0]*point_vec[1] - edge_vec[1]*point_vec[0]'
    substituteInPlace "$out/pygarment/pattern/core.py" \
      --replace-fail 'np.cross(first_edge, last_edge)' '(first_edge[0]*last_edge[1] - first_edge[1]*last_edge[0])'
    substituteInPlace "$out/pygarment/garmentcode/operators.py" \
      --replace-fail 'right_position = np.sign(np.cross(insert_vector, avg_vertex - np.asarray(new_edges[0].start))) == -1' \
      'right_position = insert_vector[0]*(avg_vertex[1]-new_edges[0].start[1]) - insert_vector[1]*(avg_vertex[0]-new_edges[0].start[0]) < 0'
    substituteInPlace "$out/pygarment/meshgen/boxmeshgen.py" \
      --replace-fail '{self.name}::WARNING::Detected edge' 'WARNING::Detected edge'
    install -m644 ${./garmentcode_prepare.py} "$out/garmentcode_prepare.py"
  '';
in
pkgs.writeShellApplication {
  name = "anyhand-garmentcode";
  text = ''
    export MPLBACKEND=Agg
    exec ${runtime}/bin/python ${scripts}/garmentcode_prepare.py "$@"
  '';
}
