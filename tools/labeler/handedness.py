"""Associate physical hands across the calibrated Frame views and over time."""

from bisect import bisect_left
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import math

import numpy as np

from calibration import load_frame
from workspace import CAMERAS, HANDS, write_json


def overlap(a, b):
    if a is None or b is None:
        return 0.0
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    common = width * height
    return common / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - common)


def correspondence(pairs, minimum_score, minimum_margin):
    """Compare the two possible one-to-one assignments of the slots."""
    same = pairs[0][0] + pairs[1][1]
    swapped = pairs[0][1] + pairs[1][0]
    margin = abs(same - swapped)
    if max(same, swapped) >= minimum_score and margin >= minimum_margin:
        return int(swapped > same), margin
    return None


def native_assignment(old, new):
    return [[overlap(a, b) for b in new] for a in old]


def hand_rays(camera, boxes):
    rays = [None, None]
    for slot, box in enumerate(boxes):
        if box is None:
            continue
        pixel = [(box[0]+box[2])/2, (box[1]+box[3])/2]
        try:
            rays[slot] = camera.world_ray(pixel)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return rays


def stereo_pair(a, b):
    if a is None or b is None:
        return 0.0, None
    o, u = a
    p, v = b
    baseline = p-o
    dot = float(u @ v)
    denominator = 1-dot*dot
    if denominator < 0.0064:
        return 0.0, None
    along_u, along_v = float(u @ baseline), float(v @ baseline)
    depth_u = (along_u - dot*along_v) / denominator
    depth_v = (dot*along_u - along_v) / denominator
    if not (0.10 <= depth_u <= 1.30 and 0.10 <= depth_v <= 1.30):
        return 0.0, None
    first, second = o + depth_u*u, p + depth_v*v
    position = (first + second) / 2
    residual = np.linalg.norm(first-second)
    # The box centres may project to different parts of a partly occluded hand.
    score = math.exp(-0.5 * (residual / 0.025)**2)
    return score, position


def spatial_assignment(old, new, elapsed):
    # Allow for hand and headset motion between two adjacent shutter times.
    scale = 0.04 + 1.4 * elapsed / 1_000_000
    return [[math.exp(-0.5 * (np.linalg.norm(a - b) / scale)**2)
             if a is not None and b is not None else 0.0 for b in new] for a in old]


def stereo_motion_link(old, new, elapsed):
    # If the hands are close together, motion and box-centre error can make
    # the other hand the nearest 3D point. Require a separated pair on at
    # least one side and reject close pairs on either side.
    separation = [np.linalg.norm(a - b) for a, b in (old, new) if a is not None and b is not None]
    if not separation or min(separation) < 0.30:
        return None
    return correspondence(spatial_assignment(old, new, elapsed), 0.57, 0.30)


class Associations:
    """Maximum-confidence forest of same-hand and opposite-hand links."""

    def __init__(self, size):
        self.parent = list(range(size))
        self.parity = [0] * size
        self.count = [1] * size

    def find(self, node):
        parent = self.parent[node]
        if parent != node:
            root, parity = self.find(parent)
            self.parent[node] = root
            self.parity[node] ^= parity
        return self.parent[node], self.parity[node]

    def connect(self, a, b, swap):
        a, parity_a = self.find(a)
        b, parity_b = self.find(b)
        if a == b:
            return
        if self.count[a] < self.count[b]:
            a, b = b, a
        self.parent[b] = a
        self.parity[b] = parity_a ^ parity_b ^ swap
        self.count[a] += self.count[b]


def resolved_slots(workspace, raw, cameras, anchors):
    """Return per-view slot corrections using pixel continuity and stereo.

    Each view has two slots. Edges bind them by which boxes describe the same
    physical hand, independently of their model labels. Calibrated 3D side is
    only an orientation vote for a track. Its identity survives a crossing.
    """
    for name in CAMERAS:
        camera = cameras[name]
        size = workspace.manifest["cameras"][name]
        if (camera.width, camera.height) != (size["width"], size["height"]):
            raise ValueError(f"{name} calibration dimensions differ from the recording")
    nodes = []
    node_groups = {}
    edges = []
    geometry_votes = []
    previous = {}
    spatial_links = 0
    for group in workspace.groups:
        sequence, time = group["sequence"], group["time_us"]
        views = {}
        for name in CAMERAS:
            boxes = [raw.get(workspace.key(sequence, name, side), {}).get("box_xyxy") for side in HANDS]
            if not any(boxes):
                continue
            node = len(nodes)
            nodes.append((sequence, name, boxes))
            views[name] = node, hand_rays(cameras[name], boxes)
            node_groups[sequence, name] = node

        positions = {node: [None, None] for node, _ in views.values()}
        for (name_a, (first, rays_a)), (name_b, (second, rays_b)) in combinations(views.items(), 2):
            fits = [[stereo_pair(a, b) for b in rays_b] for a in rays_a]
            scores = [[score for score, _ in row] for row in fits]
            link = correspondence(scores, 0.72, 0.48)
            if link is None:
                continue
            swap, margin = link
            # A stereo correspondence and temporal overlap both constrain the
            # same slots. Give close, two-hand temporal matches precedence.
            edges.append((margin * 0.68, first, second, swap))
            # Use the front pair consistently across time. Upper views often
            # see a different part of the hand as it enters or leaves the image.
            if (name_a, name_b) == ("slam_left", "slam_right"):
                for slot in range(len(HANDS)):
                    other = slot ^ swap
                    score, point = fits[slot][other]
                    if score >= 0.70:
                        positions[first][slot] = positions[second][other] = point
            (score_a, a), (score_b, b) = fits[0][swap], fits[1][1 ^ swap]
            if min(score_a, score_b) < 0.70:
                continue
            if min(a[0], b[0]) < -0.065 and max(a[0], b[0]) > 0.065 and max(a[2], b[2]) < -0.10:
                geometry_votes.append((sequence, first, int(a[0] > b[0]), min(score_a, score_b)))

        for name, (node, _) in views.items():
            old = previous.get(name)
            previous[name] = time, node, positions[node]
            if old is None:
                continue
            old_time, old_node, old_positions = old
            elapsed = time - old_time
            if not 0 < elapsed <= 225_000:
                continue
            adjacent = elapsed <= 145_000
            paired = native_assignment(nodes[old_node][2], nodes[node][2])
            link = correspondence(paired, 0.42 if adjacent else 0.70, 0.28)
            weight = 1 if adjacent else 0.65
            if link is None and adjacent:
                link = stereo_motion_link(old_positions, positions[node], elapsed)
                weight = 0.62
                spatial_links += link is not None
            if link is not None:
                swap, margin = link
                edges.append((margin * weight, old_node, node, swap))

    tracks = Associations(len(nodes))
    for _, a, b, swap in sorted(edges, reverse=True):
        tracks.connect(a, b, swap)
    manual_votes = anchored_tracks(workspace, anchors, node_groups, nodes, tracks)
    geometry_by_time = defaultdict(float)
    for sequence, node, swap, weight in geometry_votes:
        root, parity = tracks.find(node)
        geometry_by_time[root, sequence] += weight if parity ^ swap else -weight
    geometry = defaultdict(float)
    for (root, _), vote in geometry_by_time.items():
        geometry[root] += max(-1., min(1., vote))

    model = defaultdict(int)
    for node, (_, _, boxes) in enumerate(nodes):
        root, parity = tracks.find(node)
        model[root] += sum(map(bool, boxes)) * (1 if parity else -1)
    orientation = {}
    for root, vote in model.items():
        manual = manual_votes[root]
        rig = geometry[root]
        # Require aggregate evidence across time before using the head-relative
        # side prior. A single straddling pose may have deliberately crossed arms.
        orientation[root] = int((manual if manual else rig if abs(rig) >= 3 else vote) > 0)
    result = {}
    for node, (sequence, name, _) in enumerate(nodes):
        root, parity = tracks.find(node)
        if parity ^ orientation[root]:
            result[sequence, name] = True
    return result, dict(views=len(nodes), associations=len(edges), tracks=len(model),
                        spatial_links=spatial_links,
                        manual_tracks=sum(v != 0 for v in manual_votes.values()),
                        geometry_tracks=sum(abs(v) >= 3 for v in geometry.values()),
                        swapped_views=len(result))


def anchored_tracks(workspace, anchors, node_groups, nodes, tracks):
    votes = defaultdict(float)
    times = [g["time_us"] for g in workspace.groups]
    for anchor in anchors:
        if (anchor.manifest["timestamps_sha256"] != workspace.manifest["timestamps_sha256"]
                or anchor.manifest["cameras"] != workspace.manifest["cameras"]):
            raise ValueError("The anchor workspace must belong to the same recording and camera layout")
        labels = anchor.load_labels()
        candidate = {}
        for group in anchor.groups:
            index = bisect_left(times, group["time_us"])
            nearest = min(range(max(0, index-1), min(len(times), index+1)),
                          key=lambda i: abs(times[i]-group["time_us"]))
            elapsed = abs(times[nearest]-group["time_us"])
            if elapsed > 55_000:
                continue
            for name in CAMERAS:
                node = node_groups.get((workspace.groups[nearest]["sequence"], name))
                if node is None:
                    continue
                manual = [labels.get(anchor.key(group["sequence"], name, side), {}) for side in HANDS]
                boxes = [label.get("box_xyxy") if label.get("source") in ("manual", "reviewed_sam", "reviewed_sam2") else None
                         for label in manual]
                link = correspondence(native_assignment(boxes, nodes[node][2]), 0.52, 0.36)
                if link is None:
                    continue
                swap, margin = link
                weight = margin * (1-elapsed/110_000)
                if node not in candidate or weight > candidate[node][0]:
                    candidate[node] = weight, swap
        for node, (weight, swap) in candidate.items():
            root, parity = tracks.find(node)
            votes[root] += weight if parity ^ swap else -weight
    return votes


def reconcile(workspace, xrservice, device, anchors):
    source = (workspace.root / "terra-review.json").read_bytes()
    raw = json.loads(source)
    cameras = load_frame(xrservice, device)
    flips, report = resolved_slots(workspace, raw, cameras, [workspace, *anchors])
    labels = {}
    for sequence, camera in flips:
        left, right = (workspace.key(sequence, camera, hand) for hand in HANDS)
        if left in raw and right in raw:
            labels[left] = dict(raw[right], handedness="geometry_reconciled")
            labels[right] = dict(raw[left], handedness="geometry_reconciled")
    write_json(workspace.root / "terra-handedness.json", dict(
        source_sha256=hashlib.sha256(source).hexdigest(), labels=labels, report=report))
    return report
