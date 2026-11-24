{ lib }:
let
  # Scene targets in the torso-fixed camera reference, in metres.
  hand =
    shoulder: elbow: wrist:
    let
      inherit (builtins) elemAt;
      finger =
        index:
        map (segment: [
          (elemAt wrist 0 + (index - 2) * 0.018)
          (elemAt wrist 1 + (segment + 1) * 0.025)
          (elemAt wrist 2 - 0.02)
        ]) (lib.range 0 3);
    in
    {
      arm = {
        shoulder_pair_m = shoulder;
        elbow_pair_m = elbow;
        wrist_pair_m = wrist;
      };
      keypoints_pair_m = [ wrist ] ++ lib.concatMap finger (lib.range 0 4);
    };
  reach = {
    left =
      hand
        [
          (-0.18)
          (-0.22)
          0.13
        ]
        [
          (-0.32)
          (-0.45)
          0.04
        ]
        [
          (-0.24)
          (-0.29)
          (-0.32)
        ];
    right =
      hand
        [
          0.18
          (-0.22)
          0.13
        ]
        [
          0.34
          (-0.43)
          0.03
        ]
        [
          0.21
          (-0.16)
          (-0.31)
        ];
  };
in
{
  records = [
    {
      id = "Forward reach";
      hands = reach;
    }
    {
      id = "Raised hand";
      hands = reach // {
        right =
          hand
            [
              0.18
              (-0.22)
              0.13
            ]
            [
              0.40
              (-0.19)
              (-0.03)
            ]
            [
              0.37
              0.09
              (-0.22)
            ];
      };
    }
    {
      id = "Rest";
      hands = { };
    }
  ];
}
