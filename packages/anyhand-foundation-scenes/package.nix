{
  lib,
  fetchurl,
  linkFarm,
}:
let
  revision = "16a04b1b57481d0cb26793eac0498975e8cf7f53";
  assets = builtins.fromJSON (builtins.readFile ./assets.json);
  hosts = {
    raw = "https://raw.githubusercontent.com";
    media = "https://media.githubusercontent.com/media";
  };
in
(linkFarm "anyhand-foundation-scenes-${builtins.substring 0 7 revision}" (
  lib.mapAttrsToList (path: asset: {
    name = "share/anyhand/foundation-scenes/${path}";
    path = fetchurl {
      url = "${hosts.${asset.host}}/mos9527/Foundation-Resources/${revision}/${path}";
      inherit (asset) hash;
    };
  }) assets
)).overrideAttrs
  (_: {
    meta = {
      description = "Nine Foundation glTF environments and their referenced assets for AnyHand";
      homepage = "https://github.com/mos9527/Foundation-Resources";
    };
  })
