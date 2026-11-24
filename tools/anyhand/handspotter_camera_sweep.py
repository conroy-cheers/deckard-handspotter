"""Build matched four-view rigs and score Handspotter across the geometry sweep."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from calibration import load_frame
from frame_camera import digest
from handspotter_cameras import load_stereo_rig, pair_alignment, unit
from handspotter_data import QUAD_STREAMS, load_render_images, save_json


FRAME_ORDER = ("upper_left", "slam_left", "slam_right", "upper_right")
STEPS = (0, 25, 50, 75, 100)


def upright_basis(forward):
    right = unit(np.cross(forward,[0.,1.,0.]))
    return np.column_stack((right,np.cross(forward,right),forward))


def with_reference_roll(destination, reference):
    reference_upright = upright_basis(reference[:3,2])
    roll = reference_upright.T@reference[:3,:3]
    result = destination.copy()
    result[:3,:3] = upright_basis(destination[:3,2])@roll
    return result


def interpolate(start, end, fraction):
    result = np.eye(4)
    result[:3,:3] = Slerp([0.,1.], Rotation.from_matrix([start[:3,:3],end[:3,:3]]))(fraction).as_matrix()
    result[:3,3] = (1-fraction)*start[:3,3]+fraction*end[:3,3]
    return result


def build_rigs(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source, _, front = load_stereo_rig(args.source_front)
    frame = load_frame(args.frame_calibration,args.frame_device)
    alignment = pair_alignment(*(frame[name].transform for name in FRAME_ORDER[1:3]))
    target = [alignment@frame[name].transform for name in FRAME_ORDER]
    if args.source_sides:
        sides = json.loads(args.source_sides.read_text())
        side_left,side_right = [np.asarray(sides[name]) for name in ("side_left","side_right")]
        side_source = {"provided_sha256":digest(args.source_sides)}
    else:
        side_left, side_right = target[0], target[3]
        side_source = dict(source="calibrated Frame side cameras")
    start = [side_left,front[0].transform,front[1].transform,side_right]
    streams = tuple(source)
    lenses = [source[streams[eye]]["calibration"] for eye in (0,0,1,1)]
    evidence = dict(source_front_sha256=digest(args.source_front),source_sides=side_source,
                    frame_calibration_sha256=digest(args.frame_calibration),frame_device_sha256=digest(args.frame_device),
                    frame_camera_order=FRAME_ORDER,
                    alignment="Front pair midpoint, baseline, and optical-axis bisector",
                    lenses="Fixed source lenses, copied from the corresponding front eye")

    def write(name, transforms, fraction=None, calibration=lenses, provenance=evidence):
        save_json(args.out/f"{name}.json",dict(stream_order=QUAD_STREAMS,fraction=fraction,evidence=provenance,
            cameras={label:dict(source_calibration=lens,pair_from_eye=matrix.tolist())
                     for label,lens,matrix in zip(QUAD_STREAMS,calibration,transforms)}))

    for step in args.steps:
        name = f"{int(step):03d}" if int(step)==step else f"{step:05.1f}"
        write(name,[interpolate(a,b,step/100) for a,b in zip(start,target)],step/100)
    translated = [a.copy() for a in start]
    for a,b in zip(translated,target):
        a[:3,3] = b[:3,3]
    write("frame-positions",translated)
    write("frame-source-roll",[with_reference_roll(b,a) for a,b in zip(start,target)])
    if args.frame_lens or args.morph_lenses:
        records = json.loads(args.frame_calibration.read_text())["cameras"]
        circles = {row["sourceCamera"]: row["imageCircleDiameterInDeg"] for row in records}
        calibration = [dict(label=name,image_width=frame[name].width,image_height=frame[name].height,
                            projection_model_type="kb4",intrinsics=frame[name].intrinsics.tolist(),
                            distortion=frame[name].distortion.tolist(),image_circle_degrees=circles[name])
                       for name in FRAME_ORDER]
        provenance = {**evidence, "lenses": "SteamOS KB4 and image circles, full sensor resized to the Handspotter window"}
        write("100-kb4",target,1.,calibration,provenance)
        if args.morph_lenses:
            provenance = {**evidence, "lenses": "Blend forward projections on the Handspotter window and interpolate angular support"}
            for step in args.steps:
                if 0<step<100:
                    fraction = step/100
                    morph = [dict(projection_model_type="lens_blend",source=source_lens,frame=frame,fraction=fraction)
                             for source_lens,frame in zip(lenses,calibration)]
                    name = f"{int(step):03d}" if int(step)==step else f"{step:05.1f}"
                    write(f"{name}-morphed",[interpolate(a,b,fraction) for a,b in zip(start,target)],
                          fraction,morph,provenance)


def measure(pixels, records, model, min_pixels, blank_side_views=False):
    labels = np.zeros((len(pixels),4,2,4))
    present = np.zeros(labels.shape[:-1], bool)
    centers, sizes, conf = [], [], []
    for image,record,boxes,presence in zip(pixels,records,labels,present):
        if blank_side_views:
            image = image.copy()
            image[[0,3]] = 0
        for camera,stream in enumerate(QUAD_STREAMS):
            if blank_side_views and camera in (0,3):
                continue
            for slot,side in enumerate(("left","right")):
                hand = record["views"][stream]["visible_hands"].get(side)
                if hand and hand["visible_pixels"]>=min_pixels:
                    boxes[camera,slot] = hand["box_xyxy"]
                    presence[camera,slot] = True
        c,s,p = model.run(["centers","sizes","scores"],{"images":image[None]})
        centers.append(c.reshape(4,2,2)*1280)
        sizes.append(s.reshape(4,2)*1280)
        conf.append(p.reshape(4,2))
    centers,sizes,conf = map(np.asarray,(centers,sizes,conf))
    confident = (conf>=.2)&(sizes>0)
    within = ((centers>=labels[...,:2])&(centers<=labels[...,2:])).all(axis=-1)
    center = confident&within
    center_error = np.linalg.norm(centers-(labels[...,:2]+labels[...,2:])/2,axis=-1)

    def box_overlap(scale):
        radius = scale*sizes[...,None]
        predicted = np.clip(np.concatenate([centers-radius,centers+radius],axis=-1),0,[1280,1024,1280,1024])
        intersection = np.maximum(0,np.minimum(predicted[...,2:],labels[...,2:])-
                                    np.maximum(predicted[...,:2],labels[...,:2])).prod(axis=-1)
        union = (labels[...,2:]-labels[...,:2]).prod(axis=-1)+(
            predicted[...,2:]-predicted[...,:2]).prod(axis=-1)-intersection
        return confident&(union>0)&(intersection>=.5*union)

    return present, center, center_error, box_overlap(.5), box_overlap(1.)


def summarize(present, center, center_error, diameter, radius):
    count = int(present.sum())
    hits = int((center&present).sum())
    complete = present.any(axis=1)
    found = (center&present).any(axis=1)
    return dict(labels=count,centers_inside=hits,center_recall=hits/count if count else None,
                median_center_error_px=float(np.median(center_error[center&present])) if hits else None,
                diameter_iou50=int((diameter&present).sum()),radius_iou50=int((radius&present).sum()),
                hand_frames_with_visible_view=int(complete.sum()),hand_frames_found=int(found.sum()))


def score_sweep(args):
    import onnx
    import onnxruntime as ort
    from handspotter_model import check_interface

    keys = [f"{step:03d}" for step in STEPS]+args.controls
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    graph = check_interface(onnx.load(args.model),decoded=True)
    model = ort.InferenceSession(graph.SerializeToString(),opts,providers=["CPUExecutionProvider"])
    outputs = {}
    reference = None
    for key in keys:
        directory = args.datasets/key
        data = json.loads((directory/"dataset.json").read_text())
        records = data["records"]
        identity = [[r[name] for name in ("id","hands","environment","lighting","upper_body")] for r in records]
        if reference is None:
            reference = identity
        if identity!=reference or data["stream_order"]!=list(QUAD_STREAMS):
            raise ValueError(f"Unmatched scenes or invalid camera slots in {directory}")
        images = load_render_images(directory)
        if images.shape!=(len(records),4,128,160) or images.dtype!=np.uint8:
            raise ValueError(f"Invalid Handspotter inputs at {directory}")
        outputs[key] = measure(images,records,model,args.min_pixels,args.blank_side_views)
        print(key,summarize(*outputs[key]),flush=True)

    shared = np.logical_and.reduce([outputs[key][0] for key in keys[:len(STEPS)]])
    results = {}
    for key,(visible,*estimates) in outputs.items():
        results[key] = dict(all_visible=summarize(visible,*estimates),
                            common_positions=summarize(shared&visible,*estimates),
                            front=summarize(visible[:,1:3],*(value[:,1:3] for value in estimates)),
                            by_camera={name:summarize(visible[:,i:i+1],*(v[:,i:i+1] for v in estimates))
                                       for i,name in enumerate(QUAD_STREAMS)})
    save_json(args.out,dict(model_sha256=digest(args.model),pose_groups=len(reference),stream_order=QUAD_STREAMS,
        confidence=.2,min_visible_pixels=args.min_pixels,iou=.5,labels="Visible hand-mask boxes",
        blank_side_views=args.blank_side_views,
        center_decode="x and y normalized by sensor width; slots left then right",
        fits="IoU at Handspotter size treated as either diameter or radius",stages=results))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    build = sub.add_parser("rigs")
    for name in ("source-front","frame-calibration","frame-device","out"):
        build.add_argument(f"--{name}",type=Path,required=True)
    build.add_argument("--source-sides",type=Path,help="Optional side_left/side_right transforms in the source front-pair frame")
    build.add_argument("--steps",type=float,nargs="+",default=list(map(float,STEPS)))
    build.add_argument("--frame-lens",action="store_true",help="Also generate the Frame endpoint with its calibrated lenses")
    build.add_argument("--morph-lenses",action="store_true",help="Generate matching lens blends for intermediate geometries")
    build.set_defaults(run=build_rigs)
    score = sub.add_parser("score")
    for name in ("datasets","model","out"):
        score.add_argument(f"--{name}",type=Path,required=True)
    score.add_argument("--controls",nargs="*",default=[])
    score.add_argument("--min-pixels",type=int,default=128)
    score.add_argument("--blank-side-views",action="store_true",help="Score the front pair with zero-filled side inputs")
    score.set_defaults(run=score_sweep)
    arguments = parser.parse_args()
    if arguments.run==build_rigs and any(not 0<=step<=100 for step in arguments.steps):
        parser.error("Rig steps must be between 0 and 100 percent")
    arguments.run(arguments)
