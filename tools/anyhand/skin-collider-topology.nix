{
  pkgs,
  three,
  cloth,
}:
let
  somaAssets = import ./soma-assets.nix { inherit pkgs; };
  python = pkgs.python3.withPackages (p: [ p.numpy ]);
in
pkgs.runCommand "anyhand-skin-collider-topology.bin"
  {
    nativeBuildInputs = [
      python
      pkgs.nodejs_24
    ];
  }
  ''
    python - ${somaAssets}/SOMA_neutral.npz <<'PY'
    import sys
    import numpy as np

    with np.load(sys.argv[1], allow_pickle=False) as model:
        model["mean"].astype("<f4", copy=False).tofile("positions.bin")
        model["triangles"].astype("<u4").tofile("indices.bin")
    PY
    mkdir -p node_modules/three
    ln -s ${three}/build node_modules/three/build
    printf '%s\n' '{"exports":"./build/three.module.js"}' > node_modules/three/package.json
    cp ${cloth}/mesh-bvh.mjs .
    cat > topology.mjs <<'JS'
    import {readFileSync, writeFileSync} from 'node:fs';
    import {BufferAttribute, BufferGeometry} from 'three';
    import {MeshBVH} from './mesh-bvh.mjs';

    const positions = readFileSync('positions.bin');
    const indices = readFileSync('indices.bin');
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(
      new Float32Array(positions.buffer, positions.byteOffset, positions.length / 4), 3));
    geometry.setIndex(new BufferAttribute(
      new Uint32Array(indices.buffer, indices.byteOffset, indices.length / 4).slice(), 1));
    const tree = new MeshBVH(geometry, {targetLeafSize: 8});
    const bvh = MeshBVH.serialize(tree, {cloneBuffers: false});
    if (bvh.roots.length !== 1) throw new Error('SOMA-X skin topology must have one BVH root');
    const root = Buffer.from(bvh.roots[0]);
    const ordered = Buffer.from(bvh.index.buffer, bvh.index.byteOffset, bvh.index.byteLength);
    const header = Buffer.alloc(16);
    header.writeUInt32LE(bvh.version, 0);
    header.writeUInt32LE(positions.length / 4, 4);
    header.writeUInt32LE(indices.length / 4, 8);
    header.writeUInt32LE(root.length, 12);
    writeFileSync(process.argv[2], Buffer.concat([header, indices, ordered, root]));
    JS
    node topology.mjs "$out"
  ''
