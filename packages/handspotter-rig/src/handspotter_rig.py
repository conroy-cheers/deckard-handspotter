"""Export the renderer's native four-camera rig from a Frame's calibration."""
import argparse
from pathlib import Path

from rig_export import export_rig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xrservice", type=Path, required=True)
    parser.add_argument("--device-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export_rig(args.xrservice, args.device_config, args.out)
