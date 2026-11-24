"""Calibrated KB4 and FISHEYE624 camera models in detector slot order."""
import json

import numpy as np

from frame_camera import rays as frame_rays
from calibration import quaternion_transform
from handspotter_data import QUAD_STREAMS, SIDES
from geometry import Camera


def unit(v):
    return v/np.linalg.norm(v)


def pair_alignment(left, right):
    baseline = right[:3, 3] - left[:3, 3]
    if not np.isfinite(baseline).all() or np.linalg.norm(baseline) < 1e-9:
        raise ValueError("Stereo calibration requires distinct camera centers")
    lateral = unit(baseline)
    forward = left[:3, 2] + right[:3, 2]
    forward -= lateral * np.dot(forward, lateral)
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("Stereo optical axes do not define a forward direction")
    forward = unit(forward)
    basis = np.stack((lateral,np.cross(lateral,forward),-forward))
    out = np.eye(4)
    out[:3,:3] = basis
    out[:3,3] = -basis@((left[:3,3]+right[:3,3])*.5)
    return out


def fisheye_radial_domain(coefficients):
    theta=np.linspace(0,np.pi/2,32769)
    radial=theta*(1+sum(c*theta**(2*i) for i,c in enumerate(coefficients,1)))
    # Use the increasing branch connected to the optical axis.
    turning=np.flatnonzero(np.diff(radial)<=0)
    end=turning[0]+1 if len(turning) else len(theta)
    return theta[:end],radial[:end]


class Fisheye624Camera:
    def __init__(self, calibration, transform, render_scale=None):
        from hand_tracking_toolkit.camera import from_json

        width, height = calibration["image_width"], calibration["image_height"]
        params = calibration["projection_params"]
        if (calibration["projection_model_type"] != "CameraModelType.FISHEYE624"
                or len(params) not in (15, 16) or min(width, height) < 1):
            raise ValueError("Expected calibrated FISHEYE624 intrinsics and positive sensor dimensions")
        self.width, self.height = (width, height) if render_scale is None else (160*render_scale, 128*render_scale)
        scale = np.array([self.width/width, self.height/height])
        intrinsics = np.asarray(params[:-12], dtype=np.float64)
        self.coefficients = np.asarray(params[-12:], dtype=np.float64)
        focal = np.broadcast_to(intrinsics[:-2], (2,)) * scale
        center = (intrinsics[-2:] + .5) * scale - .5
        if not np.isfinite(params).all() or np.any(focal <= 0):
            raise ValueError("Expected finite FISHEYE624 parameters and positive focal lengths")
        resized = dict(calibration, image_width=self.width, image_height=self.height,
                       serial_number=calibration.get("serial_number", ""), label=calibration.get("label", ""),
                       projection_params=[*focal, *center, *self.coefficients])
        identity = dict(quaternion_wxyz=[1., 0., 0., 0.], translation_xyz=[0., 0., 0.])
        self.camera = from_json(dict(calibration=resized, T_world_from_camera=identity))
        self.transform, self.origin = transform, transform[:3, 3]

    def angular_pixels(self, angular):
        return self.camera.distort.evaluate(angular) * self.camera.f + self.camera.c

    @property
    def angular_limit(self):
        theta, _ = fisheye_radial_domain(self.coefficients[:6])
        return theta[-1]

    @property
    def focal(self):
        return np.asarray(self.camera.f)

    def rays(self, nominal_grid=None):
        """Invert FISHEYE624, retaining the complete sensor window within 90°."""
        if nominal_grid is None:
            y,x=np.indices((self.height,self.width),dtype=np.float64)
        else:
            x,y=nominal_grid
        distorted=(np.stack((x,y),axis=-1)-self.camera.c)/self.camera.f
        k=self.coefficients[:6]
        p1,p2,s1,s2,s3,s4=self.coefficients[6:]
        normalized=distorted.copy()
        for _ in range(6):
            u,v=np.moveaxis(normalized,-1,0)
            r2=u*u+v*v
            tangential=np.stack((
                2*p2*u*v+p1*(r2+2*u*u)+s1*r2+s2*r2*r2,
                2*p1*u*v+p2*(r2+2*v*v)+s3*r2+s4*r2*r2,
            ),axis=-1)
            normalized=distorted-tangential
        radius=np.linalg.norm(normalized,axis=-1)
        theta,radial=fisheye_radial_domain(k)
        angle=np.interp(radius,radial,theta)
        scale=np.divide(np.sin(angle),radius,out=np.ones_like(radius),where=radius>0)
        rays=np.dstack((normalized*scale[...,None],np.cos(angle)))
        valid=radius<=radial[-1]
        return rays,valid,float(np.degrees(theta[-1]))

    def project(self,points):
        eye=(points-self.origin)@self.transform[:3,:3]
        return self.camera.eye_to_window(eye),eye[:,2]

    def lookup(self, raster, lens=None):
        rays,valid,angle_limit=self.rays(lens.nominal_grid(self.width,self.height) if lens else None)
        expected=np.stack(np.where(valid)[::-1],axis=-1)
        forward=self.camera.eye_to_window(rays[valid])
        if lens:
            forward=lens.project(forward,self.width)
        maximum=float(np.abs(expected-forward).max())
        if maximum>.0001:
            raise ValueError(f'FISHEYE624 inversion mismatch {maximum} px')
        return raster.lookup_rays(self,rays,valid),dict(max_roundtrip_pixels=maximum,valid_fraction=float(valid.mean()),max_rendered_incidence_deg=angle_limit)


class FrameHandspotterCamera:
    """Frame camera with the full native ROI mapped into the Handspotter window."""

    def __init__(self, calibration, transform, render_scale=8):
        native_width = calibration["image_width"]
        native_height = calibration["image_height"]
        self.camera = Camera(calibration["label"], native_width, native_height, "kb4",
                             calibration["intrinsics"], calibration["distortion"], transform)
        self.circle = calibration["image_circle_degrees"]
        self.width, self.height = ((native_width, native_height) if render_scale is None
                                   else (160*render_scale, 128*render_scale))
        self.scale = np.array([self.width/native_width, self.height/native_height])
        self.transform, self.origin = transform, self.camera.origin

    def angular_pixels(self, angular):
        r2 = np.sum(angular*angular, axis=-1)
        theta = angular * np.polynomial.polynomial.polyval(r2, np.r_[1., self.camera.distortion[:4]])[..., None]
        native = theta * self.camera.f + self.camera.c
        return (native+.5) * self.scale-.5

    @property
    def angular_limit(self):
        return np.deg2rad(self.circle/2)

    @property
    def focal(self):
        return self.scale * self.camera.f

    def project(self, points):
        eye = (points-self.origin)@self.transform[:3,:3]
        native = self.camera.project_eye(eye)
        return (native+.5)*self.scale-.5, eye[:,2]

    def lookup(self, cube, lens=None):
        grid = lens.nominal_grid(self.width, self.height) if lens else None
        rays, valid = frame_rays(self.camera, self.circle, self.height, self.width, grid)
        forward = (self.camera.project_eye(rays[valid])+.5)*self.scale-.5
        if lens:
            forward = lens.project(forward, self.width)
        error = float(np.abs(forward-np.stack(np.where(valid)[::-1],axis=-1)).max())
        if error>1e-4:
            raise ValueError(f"KB4 inversion mismatch {error} px")
        return cube.lookup_rays(self,rays,valid), dict(
            max_roundtrip_pixels=error, valid_fraction=float(valid.mean()),
            max_rendered_incidence_deg=self.circle/2,
        )


class BlendedHandspotterCamera:
    """Blend the two lenses' forward projections in a common pixel window."""

    def __init__(self, source, frame, fraction):
        if not 0<fraction<1:
            raise ValueError("Lens blends require an intermediate fraction")
        self.source, self.frame, self.fraction = source, frame, fraction
        self.width, self.height = frame.width, frame.height
        self.transform, self.origin = frame.transform, frame.origin
        self.limit = (1-fraction)*source.angular_limit + fraction*frame.angular_limit
        self.center = self.from_angles(np.zeros((1,2)))[0]
        self.focal = (1-fraction)*source.focal + fraction*frame.focal

    def from_angles(self, angular):
        return (1-self.fraction)*self.source.angular_pixels(angular) + self.fraction*self.frame.angular_pixels(angular)

    def project(self, points):
        eye = (points-self.origin)@self.transform[:3,:3]
        radius = np.linalg.norm(eye[..., :2], axis=-1, keepdims=True)
        angle = np.arctan2(radius, eye[..., 2:])
        angular = eye[..., :2] * np.divide(angle, radius, out=np.ones_like(radius), where=radius!=0)
        return self.from_angles(angular), eye[:,2]

    def radial_domain(self):
        # The support is the projection of the angular circle through the
        # blended lens. Check that it is star-shaped about the optical axis.
        azimuth = np.linspace(0,2*np.pi,8193)
        rim = self.from_angles(self.limit*np.stack((np.cos(azimuth),np.sin(azimuth)),axis=-1))-self.center
        bearing = np.unwrap(np.arctan2(rim[:,1],rim[:,0]))
        if np.any(np.diff(bearing)<=0):
            raise ValueError("Blended image circle folds over")
        theta = np.linspace(0,self.limit,8193)
        axes = (self.from_angles(np.stack((theta,np.zeros_like(theta)),axis=-1))[:,0]-self.center[0])/self.focal[0]
        if np.any(np.diff(axes)<=0):
            raise ValueError("Blended radial lookup folds over")
        return bearing, rim, axes, theta

    def lookup(self, cube, lens=None):
        bearing, rim, axes, theta = self.radial_domain()
        rays = np.zeros((self.height,self.width,3),np.float64)
        valid = np.zeros(rays.shape[:2],bool)
        worst = 0.
        u = np.arange(self.width)
        grid = lens.nominal_grid(self.width, self.height) if lens else None
        for start in range(0,self.height,96):
            rows = slice(start,min(start+96,self.height))
            x,y = (grid[0][rows], grid[1][rows]) if grid else np.meshgrid(u,np.arange(rows.start,rows.stop))
            offset = np.stack((x,y),axis=-1)-self.center
            phi = np.arctan2(offset[...,1],offset[...,0])
            support = np.interp(phi,bearing,np.linalg.norm(rim,axis=1),period=2*np.pi)
            inside = np.linalg.norm(offset,axis=-1)<=support
            destination = offset[inside]+self.center
            goal = offset[inside]/self.focal
            radius = np.linalg.norm(goal,axis=-1)
            angular = goal*np.divide(np.interp(radius,axes,theta),radius,
                                     out=np.ones_like(radius),where=radius>0)[:,None]
            # A bounded Newton step handles the anisotropic resize,
            # tangential distortion and prism terms of the blended mapping.
            for _ in range(16):
                pixel = self.from_angles(angular)
                error = pixel-destination
                if np.abs(error).max(initial=0)<1e-5:
                    break
                dx = (self.from_angles(angular+[1e-5,0])-pixel)*1e5
                dy = (self.from_angles(angular+[0,1e-5])-pixel)*1e5
                jacobian = np.stack((dx,dy),axis=-1)
                if np.any(np.linalg.det(jacobian)<=0):
                    raise ValueError("Non-invertible blended lens Jacobian")
                step = np.linalg.solve(jacobian,error[...,None])[...,0]
                angular -= step/np.maximum(1.,np.linalg.norm(step,axis=-1,keepdims=True)/.1)
                angle = np.linalg.norm(angular,axis=-1,keepdims=True)
                angular /= np.maximum(1.,angle/(self.limit+1e-6))
            projected = self.from_angles(angular)
            if lens:
                projected = lens.project(projected, self.width)
                xy = np.meshgrid(u,np.arange(rows.start,rows.stop))
                destination = np.stack(xy,axis=-1)[inside]
            residual = float(np.abs(projected-destination).max(initial=0))
            worst = max(worst,residual)
            valid[rows] = inside
            angle = np.linalg.norm(angular, axis=-1, keepdims=True)
            rays[rows][inside] = np.concatenate((angular*np.sinc(angle/np.pi), np.cos(angle)), axis=-1)
        if worst>1e-4:
            raise ValueError(f"Blended lens inversion mismatch {worst} px")
        return cube.lookup_rays(self,rays,valid), dict(
            max_roundtrip_pixels=worst, valid_fraction=float(valid.mean()),
            max_rendered_incidence_deg=float(np.degrees(self.limit)),
        )


def rig_camera(calibration, transform, render_scale):
    model = calibration["projection_model_type"]
    if model == "kb4":
        return FrameHandspotterCamera(calibration, transform, render_scale)
    if model == "CameraModelType.FISHEYE624":
        return Fisheye624Camera(calibration, transform, render_scale)
    if model == "lens_blend":
        # Older sweep files named the source endpoint after the capture device.
        # Identify that endpoint by its recorded projection model.
        source = calibration.get("source")
        if source is None:
            source, = [v for v in calibration.values() if isinstance(v, dict)
                       and v.get("projection_model_type") == "CameraModelType.FISHEYE624"]
        if source["projection_model_type"] not in ("kb4", "CameraModelType.FISHEYE624"):
            raise ValueError("Expected a calibrated KB4 or FISHEYE624 source lens")
        # Blends live in the common detector window, including in stereo rigs.
        scale = 8 if render_scale is None else render_scale
        frame = FrameHandspotterCamera(calibration["frame"], transform, scale)
        lens = rig_camera(source, transform, scale)
        return BlendedHandspotterCamera(lens, frame, calibration["fraction"])
    raise ValueError(f"Unsupported rig projection model {model!r}")


def rigid_transform(values, name):
    world = np.asarray(values, np.float64)
    if world.shape != (4, 4) or not np.isfinite(world).all() or not np.allclose(world[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid rigid transform for {name}")
    rotation = world[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ValueError(f"Invalid rotation for {name}")
    return world


def cameras_from_rig(records, streams, render_scale):
    source, cameras = {}, []
    for name in streams:
        record = records["cameras"][name]
        world = rigid_transform(record["pair_from_eye"], name)
        calibration = record["source_calibration"]
        cameras.append(rig_camera(calibration, world, render_scale))
        source[name] = dict(calibration=calibration)
    pair_from_device = rigid_transform(records.get("pair_from_device", np.eye(4)), "pair_from_device")
    return source, pair_from_device, cameras


def load_quad_rig(path, render_scale=8):
    records = json.loads(path.read_text())
    if records["stream_order"] != list(QUAD_STREAMS):
        raise ValueError(f"Expected Handspotter slot order {QUAD_STREAMS}")
    return cameras_from_rig(records, QUAD_STREAMS, render_scale)


def stereo_streams(records):
    """Select left and right tracking views by calibration label or stream name."""
    streams = []
    for side in SIDES:
        matches = []
        for name, record in records.items():
            label = record["calibration"].get("label", "").lower().replace("_", "-")
            aliases = (side, f"front-{side}", f"slam-{side}", f"camera-slam-{side}")
            if name.lower().replace("_", "-") in aliases or label in aliases:
                matches.append(name)
        if len(matches) != 1:
            raise ValueError(f"Expected one {side} tracking camera; supply a rig with an explicit stream_order")
        streams.extend(matches)
    if streams[0] == streams[1]:
        raise ValueError("Left and right must be different cameras")
    return tuple(streams)


def load_stereo_rig(path):
    records = json.loads(path.read_text())
    if "stream_order" in records:
        streams = records["stream_order"]
        if len(streams) != 2 or len(set(streams)) != 2 or set(records["cameras"]) != set(streams):
            raise ValueError("A stereo rig needs two cameras in left-to-right stream_order")
        return cameras_from_rig(records, streams, None)
    streams = stereo_streams(records)
    source = {name: records[name] for name in streams}
    lenses = [row["calibration"] for row in source.values()]
    poses = [quaternion_transform(cal["T_device_from_camera"]) for cal in lenses]
    alignment = pair_alignment(*poses)
    cameras = [rig_camera(cal, alignment @ pose, None) for cal, pose in zip(lenses, poses, strict=True)]
    return source, alignment, cameras
