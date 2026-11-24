"""Fit patterned upper-body garment shells to rest and posed SOMA-X meshes."""
from dataclasses import dataclass
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_geometry import SIDES
from garmentcode_fit import fit_compiled_garment
from replay_surface import vertex_normals


@dataclass(frozen=True)
class Garment:
    sleeve: float  # 0 at the shoulder, 1 at the elbow, 2 at the wrist
    ease: float    # radial fabric allowance in metres
    cuff: float    # radial allowance at the opening
    shape: str
    palm: float    # sleeve extension past the wrist in metres
    roughness: float
    pattern: int
    stiffness: float
    neckline: float = 0.


GARMENTS = {
    "tank-top":             Garment(.08,.018,.016,"straight",0.,.74,0,.27,-.055),
    "short-sleeve-jersey":  Garment(.48,.021,.018,"straight",0.,.67,3,.23,-.022),
    "rolled-up-plaid":      Garment(.84,.026,.024,"rolled",   0.,.84,1,.39,-.015),
    "fitted-knit":          Garment(1.87,.014,.009,"straight",0.,.94,2,.37),
    "hoodie":               Garment(1.92,.039,.014,"gathered",0.,.83,0,.29),
    "wide-bell-sleeve":     Garment(1.92,.070,.070,"flared",  0.,.71,0,.13,-.022),
    "bishop-sleeve":        Garment(1.97,.054,.012,"gathered",0.,.87,2,.19,-.013),
    "overlong-sweatshirt":  Garment(1.96,.037,.025,"gathered",.046,.84,0,.27),
    "wide-overlong-sleeve": Garment(1.96,.062,.055,"flared",  .070,.80,1,.17,-.017),
}


def clip_shell(rest, posed, normals, faces, limits, cell_size=.016):
    """Clip a skin template at the garment openings and remesh it for cloth."""
    keep = (limits[faces].max(axis=1)>=0).all(axis=1)
    lookup = {}
    samples, triangles = [], []
    data = np.column_stack((rest,posed,normals,limits))
    for face in faces[keep]:
        polygon = list(data[face])
        for plane in range(9,data.shape[1]):
            if not polygon:
                break
            clipped = []
            a = polygon[-1]
            for b in polygon:
                if (a[plane]>=0)!=(b[plane]>=0):
                    clipped.append(a+(b-a)*(a[plane]/(a[plane]-b[plane])))
                if b[plane]>=0:
                    clipped.append(b)
                a = b
            polygon = clipped
        if len(polygon)<3:
            continue
        indices = []
        for point in polygon:
            # Preserve exact cuff cuts so each boundary can be extended as a ring.
            cuff = tuple(np.abs(point[-2:])<1e-6)
            key = (*np.rint(point[:3]/cell_size).astype(int),*cuff)
            idx = lookup.setdefault(key,len(samples))
            if idx==len(samples):
                samples.append([])
            samples[idx].append(point)
            indices.append(idx)
        triangles.extend((indices[0],indices[i],indices[i+1]) for i in range(1,len(indices)-1)
                         if len({indices[0],indices[i],indices[i+1]})==3)
    vertices = np.asarray([np.mean(group,axis=0) for group in samples])
    triangles = np.asarray(triangles,dtype=np.int32)
    _,same = np.unique(np.sort(triangles,axis=1),axis=0,return_index=True)
    triangles = triangles[np.sort(same)]
    used,new = np.unique(triangles,return_inverse=True)
    return vertices[used],new.reshape(-1,3).astype(np.int32)


def unit(vec):
    return vec/np.linalg.norm(vec)


def sleeve_rings(points, faces, cuff, hand, cover):
    """Extend an open sleeve past the wrist without copying finger geometry."""
    edges = [(int(a),int(b)) for face in faces for a,b in
             zip(face,np.roll(face,-1),strict=True) if cuff[a] and cuff[b]]
    counts = Counter(tuple(sorted(edge)) for edge in edges)
    edges = np.asarray([edge for edge in edges if counts[tuple(sorted(edge))]==1])
    if len(edges)<8:
        raise ValueError("Could not find the open SOMA-X sleeve boundary")
    ring,inverse = np.unique(edges,return_inverse=True)
    rim = points[ring]
    center = rim.mean(axis=0)
    wrist = np.asarray(hand["arm"]["wrist_pair_m"])
    elbow = np.asarray(hand["arm"]["elbow_pair_m"])
    forward = unit(wrist-elbow)
    landmarks = np.asarray(hand["keypoints_pair_m"])
    palm = unit(landmarks[[5,9,13,17]].mean(axis=0)-wrist)
    axis = np.cross(forward,palm)
    angle = np.arctan2(np.linalg.norm(axis),np.dot(forward,palm))
    rotation = Rotation.from_rotvec((unit(axis)*angle*.65) if angle>1e-6 else np.zeros(3))
    gap = max(0.,float(np.dot(wrist-center,forward)))
    reach = gap+cover
    steps = max(2,int(np.ceil(reach/.016)))
    base = ring[inverse].reshape(-1,2)
    new_points, new_faces = [], []
    for i in range(1,steps+1):
        fraction = i/steps
        distance = fraction*reach
        target = center+min(gap,distance)*forward+max(0.,distance-gap)*rotation.apply(forward)
        tilt = Rotation.from_rotvec(rotation.as_rotvec()*fraction)
        new_points.append(target+tilt.apply(rim-center)*(1+.12*fraction))
        next_edges = inverse.reshape(-1,2)+len(points)+(i-1)*len(ring)
        a,b = base.T
        c,d = next_edges.T
        new_faces.append(np.column_stack((b,a,c)))
        new_faces.append(np.column_stack((b,c,d)))
        base = next_edges
    return np.concatenate(new_points),np.concatenate(new_faces)


def fit_garment(bank, record, name, rng, library=None):
    if library and name in library:
        return fit_compiled_garment(library[name], bank, record)
    design = GARMENTS[name]
    rest, posed, skin_faces = (bank[k] for k in ("rest_vertices","skin_vertices","skin_faces"))
    hips,_,_,neck = bank["rest_cloth_joints"]
    shoulder,elbow,wrist = bank["rest_arm_joints"].transpose(1,0,2)
    length = min(1.97,design.sleeve+rng.uniform(-.045,.045))
    cuffs = (shoulder[:,0]+length*(elbow[:,0]-shoulder[:,0]) if length<=1. else
             elbow[:,0]+(length-1)*(wrist[:,0]-elbow[:,0]))
    x = rest[:,0]
    collar = neck[1]+.020+design.neckline+.15*np.minimum(((x-neck[0])/.18)**2,1.)
    hem = hips[1]+rng.uniform(.025,.075)
    limits = np.column_stack((rest[:,1]-hem,collar-rest[:,1],x-cuffs[0],cuffs[1]-x))
    normals = vertex_normals(posed,skin_faces)
    nodes,faces = clip_shell(rest,posed,normals,skin_faces,limits)
    reference,positions,normal = (nodes[:,start:start+3] for start in (0,3,6))
    normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-7)

    x = reference[:,0]
    depth_from_top = nodes[:,10]
    pin = np.where((depth_from_top<.016)&(np.abs(x-neck[0])<.12),1.,0.)
    shoulder_top = (reference[:,1]>shoulder[:,1].mean()+.015)&(np.abs(x-neck[0])<.215)
    pin = np.maximum(pin,np.where(shoulder_top,.78,0.))
    # Sleeve fullness, cuff taper and material all vary independently of skin color.
    arm = np.where(x<neck[0],0,1)
    along = np.clip((x-shoulder[arm,0])/(cuffs[arm]-shoulder[arm,0]),0,1)
    ease = design.ease*rng.uniform(.88,1.12)
    cuff = design.cuff*rng.uniform(.88,1.12)
    if design.shape=="flared":
        allowance = .017+(ease-.017)*along**1.6
    elif design.shape=="gathered":
        allowance = np.interp(along,(0,.67,.86,1),(.018,ease,ease,cuff))
    elif design.shape=="rolled":
        allowance = np.interp(along,(0,.80,.90,1),(.015,ease,cuff+.012,cuff))
    else:
        allowance = np.interp(along,(0,.72,1),(ease,ease,cuff))
    offset = (allowance*(1-.55*pin))[:,None]
    rest_shell = reference + vertex_normals(reference, faces)*offset
    positions = (positions+normal*offset).astype(np.float32)

    cover = design.palm*rng.uniform(.86,1.11)
    if cover:
        # The second extension is indexed against the first extension too.
        for side,field in zip(SIDES,(-2,-1),strict=True):
            eligible = np.pad(np.abs(nodes[:,field])<1e-6,(0,len(positions)-len(nodes)))
            vertices,extension = sleeve_rings(positions,faces,eligible,record["hands"][side],cover)
            positions = np.concatenate((positions,vertices))
            faces = np.concatenate((faces,extension))
            pin = np.pad(pin,(0,len(vertices)),constant_values=0.)

    if len(positions)<400 or not np.isfinite(positions).all() or not (pin==1).any():
        raise ValueError(f"Invalid fit for {name}")
    # The garment keeps its neutral dimensions when elbows and wrists bend.
    # Cuff extensions are authored along the posed wrist beyond the base shell.
    edges = np.unique(np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(positions[edges[:,0]]-positions[edges[:,1]], axis=1)
    base = edges[:,1] < len(rest_shell)
    lengths[base] = np.linalg.norm(rest_shell[edges[base,0]]-rest_shell[edges[base,1]], axis=1)
    return dict(position=positions.ravel().tolist(),faces=faces.ravel().tolist(),
                edges=edges.tolist(), rest_lengths=lengths.tolist(),
                pin=pin.tolist(),stiffness=design.stiffness,roughness=design.roughness,
                pattern=design.pattern,coverage_mm=float(cover*1000))
