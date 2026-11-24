{ pkgs }:

pkgs.runCommand "anyhand-skin-collider.wasm"
  {
    nativeBuildInputs = [
      pkgs.llvmPackages_21.clang-unwrapped
      pkgs.llvmPackages_21.lld
    ];
  }
  ''
    clang --target=wasm32 -O3 -ffreestanding -fno-math-errno -nostdlib -fuse-ld=lld \
      -Wl,--no-entry,--export=__heap_base,--export=refit,--export=correct_anchors \
      -o "$out" ${./skin-collider.c}
  ''
