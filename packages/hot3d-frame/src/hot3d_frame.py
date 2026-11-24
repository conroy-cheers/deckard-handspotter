"""Map the labeled HOT3D tracking pair into Frame's Handspotter front views."""
import argparse
import json
from pathlib import Path
import sys

import cv2

from hot3d_warp import prepare_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "rig", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--toolkit", type=Path, help="Override the installed hand tracking toolkit")
    args = parser.parse_args()
    if args.toolkit:
        sys.path.insert(0, str(args.toolkit.resolve()))
    cv2.setNumThreads(1)
    record = prepare_clip(args.archive, args.rig.resolve(), args.out)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
