"""Frame calibration export and KB4 ray lookup."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipeline'))
from calibration import load_frame

NAMES=('slam_right','slam_left','upper_left','upper_right')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(xrservice,device):
    cameras=load_frame(xrservice,device)
    records={r['sourceCamera']:r for r in json.loads(Path(xrservice).read_text())['cameras']}
    result={}
    for name in NAMES:
        camera=cameras[name]
        raw=records[name]
        axis=camera.transform[:3,2]
        result[name]=dict(
            size=[camera.width,camera.height], camera_center_mm=(camera.origin*1000).tolist(),
            optical_axis=axis.tolist(), parent_from_eye=camera.transform.tolist(),
            yaw_deg=float(np.degrees(np.arctan2(axis[0],-axis[2]))),
            pitch_up_deg=float(np.degrees(np.arcsin(axis[1]))),
            image_circle_degrees=raw['imageCircleDiameterInDeg'], crop_region=raw.get('cropRegion'),
            kb4=next(m for m in raw['intrinsics'] if m['cameraModel']=='kb'),
            kb_spline=next(m for m in raw['intrinsics'] if m['cameraModel']=='kb_spline'),
        )
    return dict(coordinates='OpenVR HMD: +X right, +Y up, -Z forward. Camera: +X right, +Y down, +Z forward.',
                source_sha256={'xrservice':digest(xrservice),'device_config':digest(device)},
                cameras=result)


def rays(camera,circle,height,width,nominal_grid=None):
    """Inverse of KB4 on its monotonic angular domain, sampled at pixel centers."""
    if nominal_grid is None:
        ys,xs=np.indices((height,width),dtype=np.float64)
    else:
        xs,ys=nominal_grid
    px=(xs+.5)*camera.width/width-.5
    py=(ys+.5)*camera.height/height-.5
    u,v=(px-camera.c[0])/camera.f[0],(py-camera.c[1])/camera.f[1]
    radius=np.hypot(u,v)
    theta=np.linspace(0,np.deg2rad(circle/2),32769)
    radial=theta*(1+sum(k*theta**(2*i) for i,k in enumerate(camera.distortion[:4],1)))
    if np.any(np.diff(radial)<=0):
        raise ValueError(f'Nonmonotonic KB4 domain for {camera.name}')
    angle=np.interp(radius,radial,theta)
    factor=np.divide(np.sin(angle),radius,out=np.ones_like(radius),where=radius>0)
    return np.stack((u*factor,v*factor,np.cos(angle)),axis=-1),radius<=radial[-1]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xrservice',type=Path,required=True)
    parser.add_argument('--device-config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.write_text(json.dumps(export(args.xrservice,args.device_config),indent=2)+'\n')
