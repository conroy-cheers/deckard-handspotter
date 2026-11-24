"""Compile GarmentCode or GarmentCodeData sewing patterns for AnyHand."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from assets.bodies.body_params import BodyParameters
from assets.garment_programs.meta_garment import MetaGarment
from pygarment.meshgen.boxmeshgen import BoxMesh

ROOT = Path(__file__).resolve().parent
PRESETS = ("tee", "overlong")


def upper_body_pattern(specification, upper_body_only):
    pattern = specification["pattern"]
    panels = pattern["panels"]
    lower = {name for name, panel in panels.items()
             if panel.get("label") in {"leg", "left_leg", "right_leg", "skirt"}}
    if lower:
        # Upper-garment records sometimes include a separate lower garment
        # for visualisation. Keep only the upper garment's sewn components.
        for stitch in pattern.get("stitches", []):
            a, b = (side["panel"] in lower for side in stitch[:2])
            if a != b and not upper_body_only:
                raise ValueError("A lower-body panel is sewn to the upper garment; use --upper-body-only to import the top")
        if lower == panels.keys():
            raise ValueError("AnyHand garment fitting supports upper-body garments")
        pattern["panels"] = {name: panel for name, panel in panels.items() if name not in lower}
        pattern["stitches"] = [stitch for stitch in pattern.get("stitches", [])
                               if not any(side["panel"] in lower for side in stitch[:2])]
    if "panel_order" in pattern:
        # AIpparel specifications share a fixed panel order padded with nulls.
        pattern["panel_order"] = [name for name in pattern["panel_order"] if name in pattern["panels"]]
    return pattern["panels"], sorted(lower)


def compile_garment(specification, body_file, output, resolution, preset, design_file, upper_body_only=False):
    if preset or design_file:
        measurements = BodyParameters(str(body_file))
        design_file = design_file or ROOT/"assets/design_params/t-shirt.yaml"
        design = yaml.safe_load(design_file.read_text())["design"]
        if preset:
            collar = design["collar"]
            collar["width"]["v"] = -.05
            collar["fc_depth"]["v"] = .18
        if preset == "overlong":
            # Symmetric styles take their values from the main sleeve design.
            sleeve = design["sleeve"]
            for key, value in (("sleeveless", False), ("length", 1.15), ("end_width", 1.35)):
                sleeve[key]["v"] = value
        specification = MetaGarment(preset or design_file.stem, measurements, design).assembly().spec
        source = "GarmentCode"
    else:
        specification = json.loads(specification.read_text())
        source = "GarmentCode specification"
    panels, omitted_panels = upper_body_pattern(specification, upper_body_only)
    # BasicPattern normalizes imported specifications to centimetres.
    units = 100
    with tempfile.TemporaryDirectory(prefix="anyhand-garmentcode-") as tmp:
        path = Path(tmp)/"garment_specification.json"
        path.write_text(json.dumps(specification))
        mesh = BoxMesh(str(path), res=resolution*units)
        mesh.load()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)/units
    faces = np.asarray(mesh.faces, dtype=np.int32)
    panel_names, panel_faces = [], []
    for name in mesh.panelNames:
        panel = mesh.panels[name]
        ids = [mesh._get_glob_ids(panel, face) for face in panel.panel_faces]
        panel_faces.extend([len(panel_names)] * sum(len(set(face)) == 3 for face in ids))
        panel_names.append(name)
    uv_faces = np.asarray(mesh.faces_with_texture, dtype=np.int32)[:, 1::2]
    uv = np.asarray(mesh.vertex_texture, dtype=np.float64)/units
    regions = np.zeros((len(vertices), 2), dtype=np.float64)
    degree = np.bincount(faces.ravel(), minlength=len(vertices))
    arm_panels = []
    for region, side in enumerate(("left", "right")):
        names = [name for name in panel_names if side in name.lower()
                 and ("sleeve" in name.lower() or panels[name].get("label") == "arm")]
        arm_panels.append(names)
        select = np.isin(np.asarray(panel_faces), [panel_names.index(name) for name in names])
        regions[:, region] = np.bincount(faces[select].ravel(), minlength=len(vertices))/np.maximum(degree, 1)
    # The upstream mesher shares simulation vertices at sewn edges but keeps
    # independent panel-space vertices and UVs for each side of the seam.
    edges = np.unique(np.sort(faces[:, (0, 1, 1, 2, 2, 0)].reshape(-1, 2), axis=1), axis=0)
    lengths = np.array([mesh.orig_lens.get(tuple(edge), units*np.linalg.norm(
        vertices[edge[1]]-vertices[edge[0]])) for edge in edges])/units
    if not len(faces) or not np.isfinite(vertices).all() or not (lengths > 0).all():
        raise ValueError("GarmentCode produced an invalid simulation mesh")
    body = yaml.safe_load(body_file.read_text())["body"]
    needed = ("height", "head_l", "shoulder_w", "arm_length", "arm_pose_angle", "waist_line")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        version=1, source=source,
        vertices=vertices.tolist(), faces=faces.tolist(), panel_names=panel_names,
        panels=panel_faces, panel_uv=uv.tolist(), uv_faces=uv_faces.tolist(),
        seams=specification["pattern"].get("stitches", []), omitted_panels=omitted_panels,
        arm_regions=regions.tolist(), arm_panels=arm_panels, edges=edges.tolist(), rest_lengths=lengths.tolist(),
        measurements={key: body[key] for key in needed},
    ), separators=(",", ":"))+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=PRESETS)
    source.add_argument("--design", type=Path, help="GarmentCode design parameter YAML")
    source.add_argument("--spec", type=Path, help="GarmentCodeData *_specification.json")
    parser.add_argument("--upper-body-only", action="store_true",
                        help="Import the top from a pattern with lower-body panels sewn on")
    parser.add_argument("--body", type=Path, help="Matching GarmentCode body measurement YAML")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resolution", type=float, default=.018, help="Simulation mesh spacing in metres")
    args = parser.parse_args()
    if not .005 <= args.resolution <= .08:
        parser.error("--resolution must be between 0.005 and 0.08 metres")
    body = args.body or ROOT/"assets/bodies/mean_all.yaml"
    compile_garment(args.spec, body, args.out, args.resolution, args.preset, args.design, args.upper_body_only)


if __name__ == "__main__":
    main()
