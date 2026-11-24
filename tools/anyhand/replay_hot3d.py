#!/usr/bin/env python3
"""Retarget AnyHand mesh geometry to HOT3D motion and replay it in calibrated tracking views."""
import argparse
import json
from pathlib import Path
import sys
from hot3d_clip import open_clip

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from frame_camera import digest
from calibration import quaternion_transform
from render import Rasterizer,read_assets,topology,unit
from handspotter_cameras import SIDES, load_stereo_rig
from replay_plate import prepare,restore
from replay_objects import decode_mask,object_geometry,object_view,recover_object
from replay_surface import attach_forearm,fit_light,skin_detail,vertex_normals,wrist_boundary

# UmeTrack has the five tips first and no thumb-base landmark.
# AnyHand puts the wrist first, followed by four landmarks per finger.
TARGET_JOINTS=np.array([5,6,7,0,8,9,10,1,11,12,13,2,14,15,16,3,17,18,19,4])
ANYHAND_JOINTS=np.r_[0,2:21]


def pair_from_world(clip,index,pair_from_device,streams):
    metadata=clip.json(f'{index:06d}.cameras.json')
    parent=[]
    for s in streams:
        cam=metadata[s]
        device=quaternion_transform(cam['calibration']['T_device_from_camera'])
        world=quaternion_transform(cam['T_world_from_camera'])
        parent.append(pair_from_device@device@np.linalg.inv(world))
    np.testing.assert_allclose(parent[0],parent[1],rtol=0,atol=2e-5)
    return parent[0]


def observations(clip,index,pair_from_device,hand_model,streams):
    import torch
    from hand_tracking_toolkit.hand_models.umetrack_hand_model import skin_landmarks,skin_vertices
    transform_to_pair=pair_from_world(clip,index,pair_from_device,streams)
    annotation=clip.json(f'{index:06d}.hands.json')
    result={}
    for side in SIDES:
        if side not in annotation:
            continue
        item=annotation[side]
        pose=item['umetrack_pose']
        transform=transform_to_pair@quaternion_transform(pose['T_world_from_wrist'])
        if side=='right':
            transform[:,0]*=-1
        transform=torch.tensor(transform,dtype=torch.float32)
        angles=torch.tensor(pose['joint_angles'],dtype=torch.float32)
        joints=skin_landmarks(hand_model,angles,transform).numpy()
        vertices=skin_vertices(hand_model,angles,transform).numpy()
        faces=hand_model.mesh_triangles.numpy()
        if side=='right':
            faces=faces[:,::-1]
        result[side]=dict(joints=joints,vertices=vertices,faces=faces.astype(np.int32),
                          world_from_pair_direction=transform_to_pair[:3,:3].T,
                          boxes=item.get('boxes_amodal',{}),visibility=item.get('visibilities_modeled',{}))
    return result


def similarity(source,target):
    center,target_center=source.mean(-2,keepdims=True),target.mean(-2,keepdims=True)
    x,y=source-center,target-target_center
    u,s,v=np.linalg.svd(x.swapaxes(-2,-1)@y)
    sign=np.ones_like(s)
    sign[...,-1]=np.linalg.det(u@v)
    rot=(u*sign[...,None,:])@v
    scale=(s*sign).sum(axis=-1)/np.square(x).sum(axis=(-1,-2))
    shift=(target_center-(center@rot)*scale[...,None,None])[...,0,:]
    return rot,scale,shift


def reflected(asset,side):
    sign=np.array([-1.,1.,1.]) if side=='left' else np.ones(3)
    return asset['vertices']*sign,asset['joints'][ANYHAND_JOINTS]*sign


def choose_assets(assets,anchors):
    result={}
    for side in SIDES:
        targets=[frame[side]['joints'][TARGET_JOINTS] for frame in anchors if side in frame]
        if not targets:
            raise ValueError(f'No development observations for {side} hand')
        def distance(asset):
            _,joints=reflected(asset,side)
            values=[]
            for target in targets:
                rot,scale,shift=similarity(joints,target)
                values.append(np.linalg.norm(joints@rot*scale+shift-target,axis=-1).mean())
            return np.mean(values)
        result[side]=min(assets,key=distance)
    return result


def retarget(asset,target,side):
    vertices,joints=reflected(asset,side)
    rot,scale,shift=similarity(joints,target)
    moved=joints@rot*scale+shift
    mesh=vertices@rot*scale+shift
    return warp_joints(mesh,moved,target)


def warp_joints(mesh,moved,target):
    # Smooth displacements keyed by annotated joints, in centimetres.
    origin=target[0]
    warp=RBFInterpolator((moved-origin)*100,target-moved,kernel='thin_plate_spline',smoothing=.08)
    return mesh+warp((mesh-origin)*100)


class PoseBank:
    """Interpolate compatible MANO meshes after matching their articulation."""

    def __init__(self,assets,side):
        points=[reflected(asset,side) for asset in assets]
        self.vertices,self.joints=map(np.stack,zip(*points))

    def mesh(self,target):
        rot,scale,shift=similarity(self.joints,target)
        joint_positions=(self.joints@rot)*scale[:,None,None]+shift[:,None,:]
        errors=np.linalg.norm(joint_positions-target,axis=-1).mean(axis=-1)
        nearest=np.argsort(errors,kind='stable')[:4]
        distance=errors[nearest]-errors[nearest[0]]
        weights=np.exp(-distance/.002)
        weights/=weights.sum()
        aligned=(self.vertices[nearest]@rot[nearest])*scale[nearest,None,None]+shift[nearest,None,:]
        warped=[warp_joints(mesh,joints,target) for mesh,joints in zip(aligned,joint_positions[nearest])]
        return np.einsum('i,ijk->jk',weights,warped)


def match_pose(observed,bank):
    for side,hand in observed.items():
        hand['matched_mesh']=bank[side].mesh(hand['joints'][TARGET_JOINTS])
    return observed


def foreground(raster,camera,lookup,observed,selected,faces,reference=False, *,
               boundaries=None,surface_light=None,normal_maps=False,with_depth=False):
    vertices=[]
    shades=[]
    triangles=[]
    parts=[]
    offset=0
    for part,side in enumerate(SIDES,1):
        if side not in observed:
            continue
        hand=observed[side]
        if reference:
            mesh=hand['vertices']
            face=hand['faces']
            texture=np.zeros(len(mesh),dtype=mesh.dtype)
            reverse=side=='right'
        else:
            asset=selected[side]
            mesh=hand.get('matched_mesh')
            if mesh is None:
                mesh=retarget(asset,hand['joints'][TARGET_JOINTS],side)
            reverse=side=='left'
            face=faces[:,::-1] if reverse else faces
            texture=asset['skin']
        if surface_light is not None:
            light,curve,materials=surface_light
            texture=materials[side][reference]
        hand_faces=len(face)
        if boundaries is not None:
            indices,edges=boundaries[reference]
            reference_wrist=hand['vertices'][boundaries[True][0]].mean(axis=0)
            axis=unit(reference_wrist-hand['joints'][5])
            edges=edges[:,::-1] if reverse else edges
            mesh,face,texture=attach_forearm(mesh,face,texture,(indices,edges),axis)
        normals=vertex_normals(mesh,face)
        if normal_maps or surface_light is not None:
            # Anchor illumination to the room as the headset moves.
            normals=normals @ hand['world_from_pair_direction'].T
        if normal_maps:
            shade=normals
        elif surface_light is not None:
            shade=np.interp(light[0]+normals@light[1:],*curve)+texture
        else:
            diffuse=np.maximum(normals@unit(np.array([-.22,.78,.59])),0)
            shade=75+75*diffuse+texture
        triangles.extend(face+offset)
        vertices.extend(mesh)
        shades.extend(shade)
        parts.extend(np.repeat([part,part+2],[hand_faces,len(face)-hand_faces]))
        offset+=len(mesh)
    geometry=(np.asarray(vertices,np.float64),np.asarray(triangles,np.int32),np.asarray(parts,np.uint8))
    attributes=np.asarray(shades,np.float64)
    if normal_maps:
        maps=[raster.foreground(camera,lookup,*geometry,channel) for channel in attributes.reshape(-1,3).T]
        return np.stack([image for image,_ in maps],axis=-1),maps[0][1]
    return raster.foreground(camera,lookup,*geometry,attributes,with_depth=with_depth)


def removal_mask(observed,stream,image_shape,padding=12):
    mask=np.zeros(image_shape,np.uint8)
    height,width=image_shape
    for hand in observed.values():
        bounds=hand['boxes'].get(stream)
        if bounds is None:
            continue
        x0,y0,x1,y1=np.array(bounds,int)
        mask[max(0,y0-padding):min(height,y1+padding),max(0,x0-padding):min(width,x1+padding)]=255
    return mask


def object_mask(objects,stream,shape,kind):
    """Decode HOT3D's one-based row-major start/length pairs."""
    mask=np.zeros(shape,bool)
    for group in objects.values():
        for item in group:
            record=item.get(kind,{}).get(stream)
            if record:
                mask |= decode_mask(record,shape)
    return mask


def estimate_tone(records):
    # Fit only a scalar contrast and exposure from dev hand-region quantiles.
    # Do not retain or sample the reference hand texture for compositing.
    source,target=(np.concatenate([r[i] for r in records]) for i in range(2))
    p=np.percentile(source,[20,50,80])
    q=np.percentile(target,[20,50,80])
    gain=float(np.clip((q[2]-q[0])/max(1,p[2]-p[0]),.45,2.2))
    offset=float(q[1]-gain*p[1])
    return gain,offset


def estimate_tone_curve(records):
    source,target=(np.concatenate([record[i] for record in records]) for i in range(2))
    quantiles=[5,20,35,50,65,80,95]
    source_levels=np.percentile(source,quantiles)
    target_levels=np.percentile(target,quantiles)
    # Quantized source shading may have duplicate levels. Keep one knot per level.
    levels,index=np.unique(source_levels,return_index=True)
    return np.stack((levels,target_levels[index]))


def plate_weight(erase,feather):
    outside=(erase==0).astype(np.uint8)
    distance=cv2.distanceTransform(outside,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    weight=1-np.clip(distance/feather,0,1)
    return weight**2*(3-2*weight)


def composite(plate,foreground,mask,tone,seed,forearm_alpha=None):
    rng=np.random.default_rng(seed)
    shaded=foreground
    if tone is not None:
        shaded=foreground*tone[0]+tone[1] if tone.ndim==1 else np.interp(foreground,*tone)
    layer=np.clip(shaded+rng.normal(0,1.6,foreground.shape),0,255)
    alpha=(mask!=0).astype(np.float32)
    if forearm_alpha is not None:
        extension=mask>2
        alpha[extension]=forearm_alpha[extension]
    premultiplied=cv2.GaussianBlur((layer*alpha).astype(np.float32),(5,5),.80)
    alpha=cv2.GaussianBlur(alpha,(5,5),.80)
    return np.rint(premultiplied+plate*(1-alpha)).astype(np.uint8)


def save(directory,name,items):
    for key,value in items.items():
        target=directory/key/f'{name}.png'
        target.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(target),value,[cv2.IMWRITE_PNG_COMPRESSION,1]):
            raise OSError(f'Cannot save {target}')


def run(args):
    import torch
    sys.path.insert(0,str(args.toolkit.resolve()))
    from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if (set(args.fit)|set(args.plate_donors))&set(args.test):
        raise ValueError('Fitting or plate donor frames overlap replay frames')
    object_occlusion=args.object_occlusion or args.object_models is not None
    source,pair_from_device,cameras=load_stereo_rig(args.calibration)
    streams = tuple(source)
    raster=Rasterizer(args.library,size=720)
    lookups=[camera.lookup(raster)[0] for camera in cameras]
    faces=topology(args.topology)
    assets=list({a['id']:a for path in args.source for a in read_assets(path,90)}.values())
    args.out.mkdir(parents=True,exist_ok=False)
    metadata={}
    with open_clip(args.clip) as clip:
        hand_model=from_json(clip.json('__hand_shapes.json__')['umetrack'])
        anchors=[observations(clip,index,pair_from_device,hand_model,streams) for index in args.fit]
        selected=choose_assets(assets,anchors)
        pose_bank={side:PoseBank(assets,side) for side in SIDES} if args.pose_bank else None
        boundaries=None
        if args.forearms:
            reference_faces=hand_model.mesh_triangles.numpy().astype(np.int32)
            boundaries=(wrist_boundary(faces),wrist_boundary(reference_faces))
        photometry=[]
        light_samples=[]
        for index,observed in zip(args.fit,anchors):
            if pose_bank is not None:
                match_pose(observed,pose_bank)
            objects=clip.json(f'{index:06d}.objects.json') if object_occlusion and args.surface_light else {}
            for stream,camera,lookup in zip(streams,cameras,lookups):
                image=clip.pixels(index,stream)
                fg,mask=foreground(raster,camera,lookup,observed,selected,faces)
                features,exact=foreground(raster,camera,lookup,observed,selected,faces,reference=True,
                                          normal_maps=args.surface_light)
                shared=(mask!=0)&(mask==exact)
                usable=cv2.erode(shared.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
                if usable.any():
                    photometry.append((fg[usable],image[usable]))
                if args.surface_light:
                    usable &= ~object_mask(objects,stream,image.shape,'masks_amodal')
                    if usable.any():
                        light_samples.append((features[usable][::16],image[usable][::16]))
        tone=estimate_tone_curve(photometry) if args.harmonize else np.asarray(estimate_tone(photometry))
        surface_light=None
        if args.surface_light:
            light,curve=fit_light(light_samples)
            materials={side:(skin_detail(asset['vertices'])+.5*asset['skin'],
                             skin_detail(hand_model.mesh_vertices.numpy())) for side,asset in selected.items()}
            surface_light=(light,curve,materials)
        donors={s:[] for s in streams}
        for index in args.plate_donors:
            observed=observations(clip,index,pair_from_device,hand_model,streams)
            objects=clip.json(f'{index:06d}.objects.json') if object_occlusion else {}
            for stream in streams:
                image=clip.pixels(index,stream)
                erase=removal_mask(observed,stream,image.shape)
                if object_occlusion:
                    # Avoid moving a donor object's old pose into the target
                    # frame. Already visible target objects are kept separately.
                    moving=object_mask(objects,stream,image.shape,'masks_amodal')
                    erase[cv2.dilate(moving.astype(np.uint8),np.ones((7,7),np.uint8))!=0]=255
                donors[stream].append(prepare(image,erase))
        for index in args.test:
            observed=observations(clip,index,pair_from_device,hand_model,streams)
            if pose_bank is not None:
                match_pose(observed,pose_bank)
            objects=clip.json(f'{index:06d}.objects.json') if object_occlusion else {}
            images=[clip.pixels(index,stream) for stream in streams]
            object_views=None
            if args.object_models is not None:
                transform=pair_from_world(clip,index,pair_from_device,streams)
                instances,geometry=object_geometry(objects,transform,args.object_models)
                object_views=[object_view(raster,camera,lookup,stream,image,instances,geometry)
                              for camera,lookup,stream,image in zip(cameras,lookups,streams,images)]
            for camera_index,(stream,camera,lookup,image) in enumerate(zip(streams,cameras,lookups,images)):
                erase=removal_mask(observed,stream,image.shape)
                modal=object_mask(objects,stream,image.shape,'masks_modal')
                interior=cv2.erode(modal.astype(np.uint8),np.ones((5,5),np.uint8))!=0
                protected=int(np.count_nonzero((erase!=0)&interior))
                stereo_pixels=0
                if object_views is not None:
                    current,other=object_views[camera_index],object_views[1-camera_index]
                    surface,material,stereo_pixels=recover_object(current,other,(erase!=0)&~interior)
                forearm_alpha=plate_weight(erase,12) if args.forearms else None
                if args.harmonize:
                    weight=plate_weight(erase,20)
                    weight[interior]=0
                    erase=(weight>0).astype(np.uint8)*255
                erase[interior]=0
                if donors[stream]:
                    plate,filled=restore(prepare(image,erase),donors[stream])
                else:
                    plate=cv2.inpaint(image,erase,5,cv2.INPAINT_TELEA)
                    filled=0
                if args.harmonize:
                    plate=np.rint(plate*weight+image*(1-weight)).astype(np.uint8)
                if object_views is not None:
                    painted=surface&(erase!=0)
                    plate[painted]=material[painted]
                result=dict(real=image,plate=plate)
                labels={}
                depth_occluded={}
                for reference,key in ((False,'anyhand'),(True,'reference_mesh')):
                    fg,mask,*depth=foreground(raster,camera,lookup,observed,selected,faces,reference,
                                              boundaries=boundaries,surface_light=surface_light,
                                              with_depth=object_views is not None)
                    occluder=interior
                    if object_views is not None:
                        # Use the old visible matte wherever no verified
                        # object surface is available for this pixel.
                        occluder=(interior&~surface)|(surface&(current.depth+.004<depth[0]))
                        depth_occluded[key]=int(np.count_nonzero((mask!=0)&occluder))
                    mask[occluder]=0
                    result[key]=composite(plate,fg,mask,None if args.surface_light else tone,
                                          index*2+camera_index,forearm_alpha)
                    if protected and object_views is None:
                        result[key][interior]=image[interior]
                    labels[key+'_mask']=np.where(mask<=2,mask,0)
                name=f'{index:06d}-{stream}'
                save(args.out,name,result|labels|{'erase':erase})
                metadata[name]=dict(boxes={s:h['boxes'][stream] for s,h in observed.items()
                                            if stream in h['boxes'] and h['visibility'].get(stream,0)>.1},
                                    erase_pixels=int((erase>0).sum()),donor_filled_pixels=filled,protected_object_pixels=protected,
                                    stereo_object_pixels=stereo_pixels,depth_occluded=depth_occluded)
            print(f'Replayed frame {index}',flush=True)
    report=dict(stream_order=streams, sensor_sizes={s: (c.width, c.height) for s,c in zip(streams,cameras)},
                source_sha256=digest(args.clip),calibration_sha256=digest(args.calibration),development_frames=args.fit,
                replay_frames=args.test,plate_donors=args.plate_donors,selected_assets={side:asset['id'] for side,asset in selected.items()},
                tone=dict(kind='quantile_curve' if args.harmonize else 'affine',values=tone.tolist()),
                surface_light=dict(world_coefficients=light.tolist(),tone_curve=curve.tolist()) if args.surface_light else None,
                forearms=args.forearms,
                pose_bank=args.pose_bank,
                object_occlusion=object_occlusion,mesh_occlusion=args.object_models is not None,
                harmonize=args.harmonize,views=metadata)
    (args.out/'replay.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('clip','calibration','toolkit','topology','library','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',type=Path,nargs='+',required=True)
    p.add_argument('--fit',type=int,nargs='+',required=True)
    p.add_argument('--test',type=int,nargs='+',required=True)
    p.add_argument('--plate-donors',type=int,nargs='*',default=[])
    p.add_argument('--object-occlusion',action='store_true')
    p.add_argument('--object-models',type=Path,help='Use tracked HOT3D GLB meshes for object occlusion and stereo repair')
    p.add_argument('--harmonize',action='store_true',help='Fit a skin tone curve and feather plate seams')
    p.add_argument('--forearms',action='store_true',help='Extend open wrist boundaries with skin geometry')
    p.add_argument('--surface-light',action='store_true',help='Fit illumination from development frame normals')
    p.add_argument('--pose-bank',action='store_true',help='Blend the closest AnyHand source poses before rendering')
    run(p.parse_args())
