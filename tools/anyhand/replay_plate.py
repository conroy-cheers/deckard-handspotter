"""Build hand-erased backplates from a calibrated clip."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlateFrame:
    image: np.ndarray
    erased: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray | None


def prepare(image,erased):
    small=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    usable=cv2.resize(255-erased,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_NEAREST)
    keypoints,descriptors=cv2.SIFT_create(nfeatures=700).detectAndCompute(small,usable)
    points=(np.asarray([k.pt for k in keypoints],np.float32)+.5)*2-.5
    return PlateFrame(image,erased,points,descriptors)


def registered(target,donor,ring):
    if target.descriptors is None or donor.descriptors is None:
        return None
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(donor.descriptors,target.descriptors,k=2)
    matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
    if len(matches)<24:
        return None
    source=np.array([donor.points[m.queryIdx] for m in matches])
    dest=np.array([target.points[m.trainIdx] for m in matches])
    warp,inlier=cv2.findHomography(source,dest,cv2.RANSAC,3.5)
    if warp is None or int(inlier.sum())<24 or inlier.mean()<.4:
        return None
    h,w=target.image.shape
    # Zero unavailable pixels in the donor before interpolation. Requiring
    # completely valid interpolation support prevents hand pixels from
    # bleeding into the replacement even at the mask boundary.
    keep=(255-donor.erased).astype(np.float32)/255.
    valid=cv2.warpPerspective(keep,warp,(w,h))>.99999
    values=cv2.warpPerspective(donor.image.astype(np.float32)*keep,warp,(w,h))
    visible=ring&valid
    if visible.sum()<300:
        return None
    delta=target.image[visible].astype(float)-values[visible]
    gain_offset=float(np.median(delta))
    residual=float(np.median(np.abs(delta-gain_offset)))
    if residual>24:
        return None
    return residual,np.rint(np.clip(values+gain_offset,0,255)).astype(np.uint8),valid


def restore(target,donors):
    if not np.any(target.erased):
        return target.image.copy(),0
    ring=(cv2.dilate(target.erased,np.ones((27,27),np.uint8))!=0)&(target.erased==0)
    aligned=[]
    for donor in donors:
        result=registered(target,donor,ring)
        if result is not None:
            aligned.append(result)
    repaired=target.image.copy()
    missing=target.erased!=0
    filled=0
    for _,image,valid in sorted(aligned,key=lambda x:x[0]):
        fill=missing&valid
        repaired[fill]=image[fill]
        missing[fill]=False
        filled+=int(fill.sum())
    if missing.any():
        repaired=cv2.inpaint(repaired,missing.astype(np.uint8),5,cv2.INPAINT_TELEA)
    return repaired,filled
