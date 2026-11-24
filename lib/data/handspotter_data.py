"""Shared Handspotter dataset decoding, training targets, and evaluation metrics."""
import hashlib
import json

import numpy as np

QUAD_STREAMS = ("side_left", "front_left", "front_right", "side_right")
SIDES = ("left", "right")
LIMIT = np.array([1280, 1024, 1280, 1024])

def iou(a, b):
    intersection = np.maximum(0, np.minimum(a[..., 2:], b[..., 2:]) -
                              np.maximum(a[..., :2], b[..., :2])).prod(axis=-1)
    union = (a[..., 2:] - a[..., :2]).prod(axis=-1) + (b[..., 2:] - b[..., :2]).prod(axis=-1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def render_image_path(directory):
    return directory / "handspotter.npy"

def load_render_images(directory):
    return np.load(render_image_path(directory), allow_pickle=False)

def synthetic_identities(records, images):
    identities = set()
    for record, image in zip(records, images, strict=True):
        for side, hand in record["hands"].items():
            asset = hand["asset"]
            # Earlier body banks used scene-local identifiers.
            for model in ("soma-body", "mpfb-body"):
                if asset == f"{model}-{record.get('id')}-{side}" and "keypoints_pair_m" in hand:
                    points = np.asarray(hand["keypoints_pair_m"], dtype="<f4")
                    asset = f"{model}-{hashlib.sha256(points.tobytes()).hexdigest()}-{side}"
                    break
            identities.add(("pose", asset))
        if not record["hands"]:
            identities.add(("empty-frame", hashlib.sha256(np.ascontiguousarray(image)).hexdigest()))
    return identities

def load_synthetic(path, split, dropout, seed):
    data = json.loads((path/"dataset.json").read_text())
    images = load_render_images(path)
    if data["stream_order"]!=list(QUAD_STREAMS) or images.shape!=(len(data["records"]),4,128,160) or images.dtype != np.uint8:
        raise ValueError(f"Invalid Handspotter dataset at {path}")
    boxes = np.zeros((len(images),4,2,4),np.float32)
    valid = np.ones((len(images),4,2),bool)
    rng = np.random.default_rng(seed)
    for i,row in enumerate(data["records"]):
        for view,name in enumerate(QUAD_STREAMS):
            for hand,side in enumerate(SIDES):
                item = row["views"][name]["visible_hands"].get(side)
                if item and item["visible_pixels"]>=128:
                    boxes[i,view,hand] = np.asarray(item["box_xyxy"], dtype=np.float32) - .5
                elif item:
                    valid[i,view,hand] = False
        if split=="training" and rng.random()<dropout:
            images[i,[0,3]] = 0
            valid[i,[0,3]] = False
    return images,boxes,valid

def targets(boxes, valid):
    wh = (boxes[...,2:]-boxes[...,:2]).clamp(min=0)
    present = (wh>0).all(dim=-1)&valid
    center = (boxes[...,:2]+boxes[...,2:])/2560
    return center,wh.prod(dim=-1).sqrt()/1280,present

def detector_metrics(center,radius,confidence,truth,valid,common=None,*,threshold=.2):
    present = np.all(truth[...,2:]>truth[...,:2],axis=-1)&valid
    accepted = (confidence>=threshold)&(radius>0)&valid
    inside = ((center>=truth[...,:2])&(center<=truth[...,2:])).all(axis=-1)
    pred = np.clip(np.concatenate((center-radius[...,None],center+radius[...,None]),axis=-1),0,LIMIT)
    hits = accepted&present&inside
    total = int(present.sum())
    error = np.linalg.norm(center-(truth[...,:2]+truth[...,2:])/2,axis=-1)
    overlap = accepted&present&(iou(pred,truth)>=.5)
    two_views = present.sum(axis=1)>=2
    scores = dict(groups=len(truth),labels=total,center_hits=int(hits.sum()),center_recall=float(hits.sum()/total) if total else None,
                  iou50=int(overlap.sum()),accepted=int(accepted.sum()),
                  empty_slot_false_positives=int((accepted&~present).sum()),
                  median_center_error_px=float(np.median(error[present])) if total else None,
                  two_view_labels=int(two_views.sum()),
                  two_view_center_hits=int((hits.sum(axis=1)>=2).sum()),
                  by_view={name:dict(labels=int(present[:,view].sum()),center_hits=int(hits[:,view].sum()),
                                    iou50=int(overlap[:,view].sum()))
                           for view,name in enumerate(QUAD_STREAMS)})
    scores["reviewed_negative_slots"] = int((valid & ~present).sum())
    scores["by_camera_and_hand"] = {
        name: {
            side: {
                "labels": int(present[:, view, hand].sum()),
                "center_hits": int(hits[:, view, hand].sum()),
                "iou50": int(overlap[:, view, hand].sum()),
                "false_positives": int((accepted[:, view, hand] & ~present[:, view, hand]).sum()),
                "reviewed_negatives": int((valid[:, view, hand] & ~present[:, view, hand]).sum()),
            }
            for hand, side in enumerate(SIDES)
        }
        for view, name in enumerate(QUAD_STREAMS)
    }
    if common is not None:
        scores["common_labels"] = int(common.sum())
        scores["common_center_hits"] = int(hits[common].sum())
        scores["common_center_recall"] = float(hits[common].mean()) if common.any() else None
        scores["front_labels"] = int(present[:,1:3].sum())
        scores["front_center_hits"] = int(hits[:,1:3].sum())
    return scores


def save_json(path, data):
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
