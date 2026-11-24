"""Explicit calibration adapters, separate from pixel orientation/resizing."""
import json
from pathlib import Path
import numpy as np
from geometry import Camera


def axes_transform(record,scale):
    x=np.asarray(record['plus_x'],np.float64);z=np.asarray(record['plus_z'],np.float64)
    T=np.eye(4);T[:3,:3]=np.column_stack([x,np.cross(z,x),z]);T[:3,3]=np.asarray(record['position'])*scale
    if not np.isfinite(T).all() or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(T[:3,:3])<0:
        raise ValueError('Invalid calibration axes')
    return T


def frame_cameras(xrservice,device):
    """Return +Z optical cameras expressed in OpenVR HMD coordinates.

    The selected runtime's occCam0FromHeadAlignment matches the composition
    below in all six components. Camera extrinsics are mm; cad_from_cal is m.
    The head CAD axes are -X,+Y,-Z. Sensor ROI already applies cropRegion;
    subtracting its x=112 again would corrupt the declared intrinsics domain.
    KB4 is selected explicitly; the optional spline model is not approximated.
    """
    cad=device.get('cv',{}).get('cad_from_cal',device)
    cad_from_cal=axes_transform(cad,1.)
    cad_from_head=axes_transform(dict(plus_x=[-1,0,0],plus_z=[0,0,-1],position=[0,0,0]),1.)
    head_from_cal=np.linalg.inv(cad_from_head)@cad_from_cal
    result={}
    for record in xrservice['cameras']:
        name=record['sourceCamera']
        if name not in ['slam_right','slam_left','upper_left','upper_right']:continue
        if record.get('preRotateClockwiseByDeg',0)!=0:raise ValueError('Unqualified camera pre-rotation')
        intrinsics=next(i for i in record['intrinsics'] if i['cameraModel']=='kb')
        T=head_from_cal@axes_transform(record['extrinsics'],.001)
        coefficients=[intrinsics[k]for k in ['k1','k2','k3','k4']]+[0.]*4
        result[name]=Camera(name,int(record['width']),int(record['height']),'kb4',
            [intrinsics[k]for k in ['fx','fy','cx','cy']],coefficients,T)
    if set(result)!=set(['slam_right','slam_left','upper_left','upper_right']):raise ValueError('Four Frame calibrations are required')
    return result


def quaternion_transform(record):
    w,x,y,z=np.asarray(record['quaternion_wxyz'],np.float64)
    norm=w*w+x*x+y*y+z*z
    if not np.isfinite(norm) or abs(norm-1)>.01:raise ValueError('Invalid camera quaternion')
    w,x,y,z=np.array([w,x,y,z])/np.sqrt(norm)
    T=np.eye(4)
    T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    T[:3,3]=record['translation_xyz'];return T


def load_frame(xrservice='/persist/xrservice.json',device='/persist/device_config.json'):
    return frame_cameras(json.loads(Path(xrservice).read_text()),json.loads(Path(device).read_text()))
