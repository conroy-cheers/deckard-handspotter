"""Calibrated KB4 camera model; metres, +Z optical camera axes."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    name: str
    width: int
    height: int
    model: str
    intrinsics: np.ndarray
    distortion: np.ndarray
    transform: np.ndarray

    def __post_init__(self):
        self.intrinsics=np.asarray(self.intrinsics,np.float64)
        self.distortion=np.asarray(self.distortion,np.float64)
        self.transform=np.asarray(self.transform,np.float64)
        if self.model!='kb4' or self.intrinsics.shape!=(4,) or self.distortion.shape!=(8,) or self.transform.shape!=(4,4):
            raise ValueError('Camera descriptor shape/model')
        if self.width<=0 or self.height<=0 or not all(np.isfinite(v).all()for v in [self.intrinsics,self.distortion,self.transform]) or np.any(self.intrinsics[:2]<=0):
            raise ValueError('Invalid camera calibration')
        if not np.allclose(self.transform[3],[0,0,0,1],atol=1e-7) or not np.allclose(self.transform[:3,:3].T@self.transform[:3,:3],np.eye(3),atol=1e-5):
            raise ValueError('Camera transform must be rigid')
        if np.any(self.distortion[4:]):raise ValueError('KB4 extra coefficients')

    @property
    def f(self):return self.intrinsics[:2]
    @property
    def c(self):return self.intrinsics[2:]
    @property
    def origin(self):return self.transform[:3,3]

    def project_eye(self,points):
        p=np.asarray(points,np.float64);radius=np.linalg.norm(p[...,:2],axis=-1,keepdims=True)
        theta=np.arctan2(radius,p[...,2:]);angular=p[...,:2]*np.divide(theta,radius,out=np.ones_like(radius),where=radius!=0)
        r2=np.minimum(np.sum(angular**2,axis=-1,keepdims=True),np.pi**2)
        k1,k2,k3,k4=self.distortion[:4]
        radial=1+r2*(k1+r2*(k2+r2*(k3+r2*k4)))
        return angular*radial*self.f+self.c

    def unproject(self,pixels):
        pixels=np.asarray(pixels,np.float64);angles=(pixels-self.c)/self.f
        def rays(a):
            theta=np.linalg.norm(a,axis=-1,keepdims=True)
            return np.concatenate([a*np.sinc(theta/np.pi),np.cos(theta)],axis=-1)
        for _ in range(20):
            projected=self.project_eye(rays(angles));residual=projected-pixels
            if np.abs(residual).max()<1e-5:return rays(angles)
            jac=np.stack([(self.project_eye(rays(angles+np.eye(2)[i]*1e-5))-projected)/1e-5 for i in range(2)],axis=-1)
            step=np.linalg.solve(jac,residual[...,None])[...,0]
            angles-=step/np.maximum(1.,np.linalg.norm(step,axis=-1,keepdims=True)/.25)
            if not np.isfinite(angles).all() or np.any(np.linalg.norm(angles,axis=-1)>2.):break
        raise ValueError('Camera inverse did not converge')

    def world_ray(self,pixel):
        direction=self.transform[:3,:3]@self.unproject(np.asarray(pixel)[None])[0]
        return self.origin,direction/np.linalg.norm(direction)

    def project(self,point):
        eye=(np.asarray(point)-self.origin)@self.transform[:3,:3]
        return self.project_eye(eye),eye[2]
