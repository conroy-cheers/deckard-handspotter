"""Build a clothed bust with openings that join the posed upper arms."""

import numpy as np


SHOULDER_INSET = .064
SIDES_PER_RING = 32
# Ring height from the shoulder joints, width relative to their spacing,
# front/back half-depth, and the back offset from the shoulder joints.
PROFILE = np.array([
    [-.48, .82, .093, .045],
    [-.36, .79, .095, .028],
    [-.26, .83, .106, .013],
    [-.18, .88, .119, .003],
    [-.125, .92, .124, .003],
    [-.075, .99, .114, .003],
    [-.024, 1.02, .091, .007],
    [ .028, .89, .069, .014],
    [ .056, .58, .061, .019],
    [ .081, .34, .052, .018],
    [ .126, .32, .050, .018],
])
SOCKET_BOTTOM, SOCKET_TOP, SOCKET_HALF_WIDTH = 4, 7, 3


def oriented_loop(edges):
    """Order boundary vertices in the winding direction of their shared faces."""
    next_vertex = dict(edges)
    loop = [int(edges[0,0])]
    for _ in range(len(edges)-1):
        loop.append(next_vertex[loop[-1]])
    return np.asarray(loop)


def loop_fractions(vertices):
    lengths = np.linalg.norm(np.roll(vertices, -1, axis=0)-vertices, axis=1)
    return np.r_[0., np.cumsum(lengths)/lengths.sum()]


def shoulder_bridge(positions, torso_loop, arm_loop):
    """Join the loops with opposite boundary winding and minimal twist."""
    torso = positions[torso_loop]
    arm = positions[arm_loop]
    a = loop_fractions(torso)
    b = loop_fractions(arm)
    arm_at_torso = np.stack([np.interp(a[:-1], b, np.r_[column, column[0]]) for column in arm.T], axis=1)
    # Evaluate each possible phase of the arm opening against the fixed socket.
    best = np.square(arm_at_torso-torso).sum()
    roll = 0
    for i in range(1,len(arm)):
        candidate = np.roll(arm, -i, axis=0)
        fractions = loop_fractions(candidate)
        aligned = np.stack([np.interp(a[:-1], fractions, np.r_[c,c[0]]) for c in candidate.T], axis=1)
        error = np.square(aligned-torso).sum()
        if error<best:
            best,roll=error,i
    arm_loop = np.roll(arm_loop,-roll)
    b = loop_fractions(positions[arm_loop])

    faces=[]
    i=j=0
    for advance_torso in np.argsort(np.r_[a[1:],b[1:]],kind="stable")<len(torso_loop):
        p,q=torso_loop[i%len(torso_loop)],arm_loop[j%len(arm_loop)]
        if advance_torso:
            i+=1
            faces.append((torso_loop[i%len(torso_loop)],p,q))
        else:
            j+=1
            faces.append((p,q,arm_loop[j%len(arm_loop)]))
    return faces


def torso_surface(body, arm_openings, clothing_girth=1.):
    """Return torso geometry and the indices shared with each arm rim."""
    def vertex(ring, column):
        return ring*SIDES_PER_RING+column%SIDES_PER_RING

    angle = np.arange(SIDES_PER_RING)*2*np.pi/SIDES_PER_RING
    height,width,depth,back = PROFILE.T
    shell = np.stack(np.broadcast_arrays(
        width[:,None]*body.half_shoulder_width*np.cos(angle),
        (body.shoulder_y+height)[:,None],
        body.shoulder_z+back[:,None]+depth[:,None]*clothing_girth*np.sin(angle),
    ),axis=-1)
    positions = list(shell.reshape(-1,3))
    faces=[]
    centers=(SIDES_PER_RING//2,0)
    holes={}
    for side in arm_openings:
        first=centers[side]-SOCKET_HALF_WIDTH
        last=centers[side]+SOCKET_HALF_WIDTH
        holes[side]=((first,last),np.array([
            *(vertex(SOCKET_BOTTOM,k) for k in range(first,last)),
            *(vertex(r,last) for r in range(SOCKET_BOTTOM,SOCKET_TOP)),
            *(vertex(SOCKET_TOP,k) for k in range(last,first,-1)),
            *(vertex(r,first) for r in range(SOCKET_TOP,SOCKET_BOTTOM,-1)),
        ]))
    cut_columns={k%SIDES_PER_RING for (first,last),_ in holes.values() for k in range(first,last)}
    for ring in range(len(shell)-1):
        for k in range(SIDES_PER_RING):
            if SOCKET_BOTTOM<=ring<SOCKET_TOP and k in cut_columns:
                continue
            a,b=vertex(ring,k),vertex(ring,k+1)
            c,d=vertex(ring+1,k),vertex(ring+1,k+1)
            faces.extend(((a,c,b),(b,c,d)))

    for ring in (0,len(shell)-1):
        cap=len(positions)
        positions.append(shell[ring].mean(axis=0))
        for k in range(SIDES_PER_RING):
            a,b=vertex(ring,k),vertex(ring,k+1)
            faces.append((a,b,cap) if ring==0 else (b,a,cap))

    seams={}
    for side,(rim,edges) in arm_openings.items():
        seam=np.arange(len(positions),len(positions)+len(rim))
        positions.extend(rim)
        seams[side]=seam
        faces.extend(shoulder_bridge(np.asarray(positions),holes[side][1],seam[oriented_loop(edges)[::-1]]))
    return np.asarray(positions),np.asarray(faces),seams
