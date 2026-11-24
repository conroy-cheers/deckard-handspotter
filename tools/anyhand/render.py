"""Read released AnyHand poses and rasterize them through calibrated Frame cameras."""
import ctypes as C
import hashlib
import io
import pickle
import tarfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from handspotter_cameras import unit
from frame_camera import rays


class AnyHandReader(pickle.Unpickler):
    """The released WebDataset metadata needs only these NumPy globals."""
    def find_class(self,module,name):
        from numpy._core import multiarray
        allowed={
            ('numpy','dtype'):np.dtype, ('numpy','ndarray'):np.ndarray,
            ('numpy.core.multiarray','_reconstruct'):multiarray._reconstruct,
            ('numpy.core.multiarray','scalar'):multiarray.scalar,
        }
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected metadata global {module}.{name}')
        return allowed[module,name]


def topology(path):
    lines=path.read_text().splitlines()
    if sum(line.startswith('v ') for line in lines)!=778:
        raise ValueError('Expected a public 778-vertex MANO topology OBJ')
    triangles=np.array([[int(x.split('/')[0])-1 for x in line.split()[1:]]
                        for line in lines if line.startswith('f ')],np.int32)
    if triangles.shape!=(1538,3) or triangles.min()<0 or triangles.max()>=778:
        raise ValueError('Unexpected MANO topology')
    return triangles


def pose_id(meta):
    # AnyHand stores multiple views of one initial shape/articulation, with a
    # different root rotation per source image. Keep all of them in one split.
    values=np.r_[meta['betas'],meta['hand_pose'][3:]].round(5)
    return hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()[:20]


def prepare_asset(meta,encoded,encoded_mask):
    joints=np.array(meta['joints_3d'],np.float64)
    vertices=np.array(meta['verts_3d'],np.float64)
    if joints.shape!=(21,3) or vertices.shape!=(778,3) or not np.isfinite(vertices).all():
        raise ValueError('Unexpected AnyHand mesh')
    image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_GRAYSCALE)
    mask=cv2.imdecode(np.frombuffer(encoded_mask,np.uint8),cv2.IMREAD_GRAYSCALE)>0
    h,w=image.shape
    f=meta['focal_length']
    pixels=vertices[:,:2]/vertices[:,2:]*[f*w/h,f]+[w/2,h/2]
    x,y=np.rint(pixels).astype(int).T
    x=x.clip(0,w-1)
    y=y.clip(0,h-1)
    skin=image[y,x].astype(np.float32)
    valid=mask[y,x]
    skin[~valid]=np.median(skin[valid]) if np.any(valid) else 140
    skin=np.clip((skin- np.median(skin))*.30,-16,16)
    background=cv2.inpaint(image,cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8)),5,cv2.INPAINT_TELEA)
    wrist=joints[0]
    up=unit(joints[9]-wrist)
    lateral=joints[5]-joints[17]
    lateral=unit(lateral-up*np.dot(up,lateral))
    local=np.column_stack((lateral,up,np.cross(lateral,up)))
    return dict(id=pose_id(meta), vertices=(vertices-wrist)@local, joints=(joints-wrist)@local,
                skin=skin,background=background)


def read_assets(path,limit):
    assets={}
    with tarfile.open(path,'r|*') as archive:
        frame={}
        try:
            for item in archive:
                if not item.isfile():
                    continue
                name,kind=item.name.split('.',1)
                if frame and name!=previous:
                    frame={}
                previous=name
                if kind in ('data.pyd','jpg','mask.png'):
                    frame[kind]=archive.extractfile(item).read()
                if len(frame)!=3:
                    continue
                meta=AnyHandReader(io.BytesIO(frame['data.pyd'])).load()[0]
                if 'verts_3d' in meta and pose_id(meta) not in assets:
                    asset=prepare_asset(meta,frame['jpg'],frame['mask.png'])
                    asset['sample']=name
                    assets[asset['id']]=asset
                    if len(assets)==limit:
                        break
        except tarfile.ReadError as error:
            if str(error)!='unexpected end of data':
                raise
            # An HTTP range prefix can end in an incomplete tar member.
    if not assets:
        raise ValueError(f'{path} supplied no released posed meshes')
    return list(assets.values())


class Cubemap:
    def __init__(self,size):
        self.size=size
        # Cubemap bases in camera axes. The calibrated support is <=180 degrees.
        self.bases=[np.identity(3)]+[Rotation.from_euler(axis,angle,degrees=True).as_matrix()
                                     for axis,angle in [('y',90),('y',-90),('x',90),('x',-90)]]

    def lookup_rays(self,camera,directions,valid):
        local=[directions@basis for basis in self.bases]
        face=np.argmax(np.stack([r[...,2] for r in local]),axis=0)
        maps=[]
        for i,ray in enumerate(local):
            mask=valid&(face==i)
            xy=ray[mask,:2]/ray[mask,2:]
            maps.append((mask,((xy+1)*(self.size-1)/2).astype(np.float32)))
        return valid,maps,directions@camera.transform[:3,:3].T


class Rasterizer(Cubemap):
    def __init__(self,library,size=192):
        super().__init__(size)
        self.lib=C.CDLL(str(library))
        self.render=self.lib.render_face
        self.render.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p]
        self.render.restype=None

    def lookup(self,camera,circle,width,captured_support=None):
        height=round(camera.height*width/camera.width)
        directions,valid=rays(camera,circle,height,width)
        if captured_support is not None:
            observed=cv2.resize(captured_support.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST)>0
            valid &= observed
        return self.lookup_rays(camera,directions,valid)

    def foreground(self,camera,lookup,vertices,faces,parts,shade,*,with_depth=False):
        """Render grayscale foreground and surface IDs on the sensor lattice."""
        valid,maps,_=lookup
        intensity=np.zeros(valid.shape,np.float32)
        part_map=np.zeros(valid.shape,np.uint8)
        if with_depth:
            distance=np.full(valid.shape,np.inf,np.float32)
        if len(faces):
            eye=(vertices-camera.origin)@camera.transform[:3,:3]
            size=self.size
            for basis,(mask,xy) in zip(self.bases,maps):
                if not len(xy):
                    continue
                values=np.ascontiguousarray(np.c_[eye@basis,shade],np.float64)
                depth=np.full((size,size),np.inf,np.float32)
                gray=np.zeros((size,size),np.float32)
                labels=np.zeros((size,size),np.uint8)
                self.render(values.ctypes.data,faces.ctypes.data,parts.ctypes.data,len(faces),size,
                            depth.ctypes.data,gray.ctypes.data,labels.ctypes.data)
                sample=np.rint(xy).astype(np.int32).clip(0,size-1)
                x,y=sample.T
                foreground=labels[y,x]
                intensity[mask]=gray[y,x]
                part_map[mask]=foreground
                if with_depth:
                    # Convert the cubemap's face-normal depth to ray distance.
                    slope=sample*(2/(size-1))-1
                    distance[mask]=depth[y,x]*np.sqrt(1+np.square(slope).sum(axis=1))
        if with_depth:
            return intensity,part_map,distance
        return intensity,part_map

    def image(self,camera,lookup,vertices,faces,parts,shade,background,yaw,profile,rng):
        valid,maps,world_rays=lookup
        # Approximate a room by six planes. Its textures and lighting stay
        # fixed across all four views, including foreground hand parallax.
        rotation=Rotation.from_euler('y',yaw).as_matrix()
        direction=world_rays@rotation
        origin=camera.origin@rotation
        bounds=np.array([2.,1.0,2.3])
        surface=np.where(direction>0,bounds,-bounds)
        t=(surface-origin)/np.where(np.abs(direction)>1e-10,direction,1e-10)
        t[t<=0]=np.inf
        walls=np.argmin(t,axis=-1)
        hit=origin+direction*np.min(t,axis=-1)[...,None]
        h,w=background.shape
        tex=np.zeros(valid.shape,np.float32)
        for i in range(3):
            mask=walls==i
            span=[axis for axis in range(3) if axis!=i]
            uv=(hit[...,span]/bounds[span]+1)*.5
            xy=np.ascontiguousarray((uv*[w-1,h-1]).astype(np.float32))
            sample=cv2.remap(background,xy[...,0],xy[...,1],cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
            tex[mask]=sample[mask]
        result=73+(tex-120)*.78+7*(world_rays[...,1]+.4)
        foreground,part_map=self.foreground(camera,lookup,vertices,faces,parts,shade)
        result=np.where(part_map!=0,foreground,result)
        # Fixed optical support plus approximate grayscale response variation.
        falloff=np.clip(1-.22*(1-(world_rays@camera.transform[:3,2])**2),0,1)
        gain,offset,blur=profile
        image=np.clip(result*falloff*gain+offset+rng.normal(0,2.5,valid.shape),0,255).astype(np.uint8)
        if blur:
            image=cv2.GaussianBlur(image,(3,3),blur)
        image[~valid]=0
        part_map[~valid]=0
        return image,part_map


def scene(rng,assets,triangles,index):
    hands=[]
    for side,probability in enumerate((.82,.84)):
        if index%9==0 or rng.random()>probability:
            continue
        part=side+1
        asset=assets[rng.integers(len(assets))]
        sign=-1 if side==0 else 1
        wrist=np.array([sign*rng.uniform(.055,.19)+rng.uniform(-.095,.095),
                        rng.uniform(-.23,-.025),-rng.uniform(.27,.52)])
        rotation=Rotation.from_euler('xyz',rng.uniform([-55,-125,-65],[45,125,65]),degrees=True).as_matrix()
        reflection=np.diag([sign,1,1])
        scale=rng.uniform(.85,1.12)
        def place(points):
            return (points@reflection*scale)@rotation.T+wrist
        vertices=place(asset['vertices'])
        joints=place(asset['joints'])
        faces=triangles if sign==1 else triangles[:,::-1].copy()
        normal=np.zeros_like(vertices)
        v=vertices[faces]
        for corner in range(3):
            np.add.at(normal,faces[:,corner],np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]))
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
        light=unit(rng.normal(size=3)+[0,1,.8])
        shade=rng.uniform(45,130)+rng.uniform(20,87)*np.maximum(normal@light,0)+asset['skin']
        shade=np.clip(shade,15,255)
        # Add a short tapered sleeve behind each wrist. Label hands separately.
        axis=unit((wrist-np.mean(joints[[5,9,13,17]],axis=0))+[0,-.23,.08])
        tangent=unit(np.cross(axis,[0.,0.,1.]))
        angle=np.arange(12)*np.pi/6
        ring=np.cos(angle)[:,None]*tangent+np.sin(angle)[:,None]*np.cross(axis,tangent)
        sleeve=np.vstack([wrist+ring*.027,wrist+axis*.24+ring*.050])
        sleeve_faces=np.array([(i,(i+1)%12,i+12) for i in range(12)]+
                              [(i+12,(i+1)%12,(i+1)%12+12) for i in range(12)],np.int32)
        hands.append(dict(vertices=np.r_[vertices,sleeve],joints=joints,side=side,asset=asset['id'],
                          faces=np.r_[faces,sleeve_faces+len(vertices)],
                          parts=np.r_[np.full(len(faces),part,np.uint8),np.full(len(sleeve_faces),3,np.uint8)],
                          shade=np.r_[shade,np.full(24,rng.uniform(35,110))]))
    all_vertices,all_faces,all_parts,all_shades=[],[],[],[]
    offset=0
    for hand in hands:
        all_vertices.extend(hand['vertices'])
        all_faces.extend(hand['faces']+offset)
        all_parts.extend(hand['parts'])
        all_shades.extend(hand['shade'])
        offset+=len(hand['vertices'])
    return (np.array(all_vertices,np.float64),np.array(all_faces,np.int32),
            np.array(all_parts,np.uint8),np.array(all_shades,np.float64),hands)
