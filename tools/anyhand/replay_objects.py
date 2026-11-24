"""Depth-aware compositing of rigid HOT3D objects, using captured grayscale."""
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from calibration import quaternion_transform


def decode_mask(record, shape):
    mask=np.zeros(shape[0]*shape[1],bool)
    if record:
        if (record['height'],record['width'])!=shape:
            raise ValueError('Object mask and sensor dimensions differ')
        for start,count in np.asarray(record['rle'],np.int64).reshape(-1,2):
            first=start-1
            last=first+count
            if first<0 or count<0 or last>len(mask):
                raise ValueError('Object mask run outside the sensor')
            mask[first:last]=True
    return mask.reshape(shape)


@lru_cache(maxsize=64)
def mesh(path):
    import trimesh

    model=trimesh.load_scene(path,skip_materials=True).to_geometry()
    return np.asarray(model.vertices,np.float64),np.asarray(model.faces,np.int32)


def object_geometry(annotations,pair_from_world,directory):
    instances=[item for key in sorted(annotations) for item in annotations[key]]
    vertices,faces,parts=[],[],[]
    offset=0
    if len(instances)>255:
        raise ValueError('Too many objects for the depth rasterizer')
    for part,item in enumerate(instances,1):
        transform=pair_from_world@quaternion_transform(item['T_world_from_object'])
        points,triangles=mesh(directory/f"obj_{int(item['object_bop_id']):06d}.glb")
        vertices.append(points@transform[:3,:3].T+transform[:3,3])
        faces.append(triangles+offset)
        parts.append(np.full(len(triangles),part,np.uint8))
        offset+=len(points)
    if not instances:
        return instances,(np.empty((0,3)),np.empty((0,3),np.int32),np.empty(0,np.uint8),np.empty(0))
    return instances,(np.concatenate(vertices),np.concatenate(faces),np.concatenate(parts),np.zeros(offset))


@dataclass
class ObjectView:
    camera: object
    rays: np.ndarray
    owner: np.ndarray
    depth: np.ndarray
    trusted: np.ndarray
    image: np.ndarray


def object_view(raster,camera,lookup,stream,image,instances,geometry):
    _,owner,depth=raster.foreground(camera,lookup,*geometry,with_depth=True)
    trusted=np.zeros(image.shape,bool)
    for part,item in enumerate(instances,1):
        rendered=owner==part
        if not rendered.any():
            continue
        amodal=decode_mask(item.get('masks_amodal',{}).get(stream),image.shape)
        modal=decode_mask(item.get('masks_modal',{}).get(stream),image.shape)
        # Restrict to the annotated object silhouette. Sampling material also
        # requires a fully interior modal pixel, away from photographed fingers.
        support=cv2.dilate(amodal.astype(np.uint8),np.ones((5,5),np.uint8))!=0
        owner[rendered&~support]=0
        safe=cv2.erode((modal&amodal).astype(np.uint8),np.ones((5,5),np.uint8))!=0
        trusted |= (owner==part)&safe
    return ObjectView(camera,lookup[2],owner,depth,trusted,image)


def recover_object(target,other,erase):
    """Reproject nearby object material from the other camera when visible."""
    ys,xs=np.where((target.owner!=0)&~target.trusted&erase)
    points=target.camera.origin+target.rays[ys,xs]*target.depth[ys,xs,None]
    pixels,z=other.camera.project(points)
    height,width=other.image.shape
    bounded=(z>0)&np.isfinite(pixels).all(axis=1)&(pixels[:,0]>=0)&(pixels[:,1]>=0)
    bounded &= (pixels[:,0]<width-1)&(pixels[:,1]<height-1)
    ys,xs,pixels,points=(x[bounded] for x in (ys,xs,pixels,points))
    x,y=np.floor(pixels).astype(int).T
    parts=target.owner[ys,xs]
    visible=np.ones(len(x),bool)
    expected=np.linalg.norm(points-other.camera.origin,axis=1)
    # Every bilinear tap must be inside the corresponding visible object
    # and on the same depth surface, even at narrow specular edges.
    for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
        at=(y+dy,x+dx)
        visible &= other.trusted[at]&(other.owner[at]==parts)&(np.abs(other.depth[at]-expected)<.012)
    pixels=pixels[visible]
    ys,xs=ys[visible],xs[visible]
    surface=target.trusted.copy()
    values=target.image.copy()
    # OpenCV remap has a signed 16-bit destination dimension limit.
    for start in range(0,len(pixels),16384):
        section=slice(start,start+16384)
        coords=pixels[section].astype(np.float32)
        values[ys[section],xs[section]]=cv2.remap(other.image,coords[:,0][None],
                                                coords[:,1][None],cv2.INTER_LINEAR).ravel()
    surface[ys,xs]=True
    return surface,values,len(pixels)
