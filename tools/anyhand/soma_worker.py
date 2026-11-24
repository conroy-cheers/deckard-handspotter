"""Native SOMA-X process for AnyHand Studio and the headless renderer."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import warnings


def serve(args, output):
    import numpy as np
    from garmentcode_fit import garment_library
    from soma_live import SomaLive
    from soma_protocol import LENGTH, MAX_PACKET, packet
    from soma_wardrobe import GARMENTS, fit_garment

    if not args.assets.is_dir() or not args.correctives.is_file():
        raise ValueError("Supply the SOMA-X asset directory and corrective checkpoint")
    rows = json.loads(args.template.read_text())["records"] if args.template else []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Sparse CSR tensor support is in beta state\.",
            category=UserWarning,
            module=r"soma\.geometry\.skeleton_transfer",
        )
        model = SomaLive(args.assets,args.correctives,args.seed,args.threads,args.rig,args.openvr)
    scene = None
    library = garment_library(args.garment_library)
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            if request["op"] == "info":
                frames = [{"id":row["id"], "angles":row.get("head_pose",{}).get("yaw_pitch_roll_deg",[0,0,0])}
                          for row in rows]
                response = packet(dict(frames=frames,cameras=model.cameras,garments=list(dict.fromkeys((*GARMENTS, *library)))),{})
            elif request["op"] == "pose":
                index = request["index"]
                row = request.get("row")
                if row is None:
                    if not 0 <= index < len(rows):
                        raise ValueError("SOMA-X scene index is outside the template")
                    row = rows[index]
                    rendered = True
                else:
                    rendered = request.get("rendered",False)
                gestures = request.get("gestures")
                seed = request.get("seed", args.seed)
                key = (seed,index,row,rendered,gestures)
                if scene != key:
                    scene = None
                    model.seed = seed
                    model.prepare(row,index,rendered,gestures)
                    scene = key
                angles = request.get("angles")
                if angles is None:
                    angles = row.get("head_pose",{}).get("yaw_pitch_roll_deg",[0,0,0])
                record, bank = model.pose(angles)
                bank["skin_faces"] = bank["skin_faces"].astype("<u4",copy=False)
                metadata = dict(record=record)
                if name := request.get("garment"):
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed,index,8123]))
                    garment = fit_garment(bank,record,name,rng,library)
                    for field,dtype in (("position","<f4"),("faces","<u4"),("pin","<f4")):
                        bank["garment_"+field] = np.asarray(garment.pop(field),dtype=dtype)
                    metadata["garment"] = garment
                response = packet(metadata,bank)
            else:
                raise ValueError(f"Unknown SOMA-X request {request['op']!r}")
            if len(response) > MAX_PACKET:
                raise ValueError("SOMA-X response exceeds the packet limit")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = packet(dict(error=str(exc)),{})
        output.write(LENGTH.pack(len(response)))
        output.write(response)
        output.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("assets","correctives"):
        parser.add_argument("--"+arg,type=Path,required=True)
    for arg in ("rig","openvr","template"):
        parser.add_argument("--"+arg,type=Path)
    parser.add_argument("--garment-library", type=Path, default=os.environ.get("ANYHAND_GARMENT_LIBRARY"))
    parser.add_argument("--seed",type=int,default=681721)
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    if bool(args.rig) != bool(args.openvr):
        parser.error("--rig and --openvr must be supplied together")
    if args.threads < 1:
        parser.error("--threads must be positive")
    # Warp and OpenUSD can write directly to stdout. Give them stderr and
    # retain the original pipe exclusively for binary responses.
    with os.fdopen(os.dup(1),"wb") as binary:
        os.dup2(2,1)
        serve(args,binary)
