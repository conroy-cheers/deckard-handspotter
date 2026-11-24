#!/usr/bin/env python3
"""Render AnyHand meshes through a calibrated stereo camera pair."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from handspotter_cameras import SIDES, load_stereo_rig
from frame_camera import digest
from render import Rasterizer, read_assets, scene, topology
from stereo_images import visible_boxes


def run(args):
    if args.toolkit:
        sys.path.insert(0,str(args.toolkit.resolve()))
    if args.count<1 or args.negative_every<0 or args.face_size<2 or args.asset_limit<1:
        raise ValueError('Invalid rendering limits')
    source,pair_from_device,cameras=load_stereo_rig(args.calibration)
    streams = tuple(source)
    assets=list({a['id']:a for path in args.source for a in read_assets(path,args.asset_limit)}.values())
    triangles=topology(args.topology)
    raster=Rasterizer(args.library,size=args.face_size)
    combined=[camera.lookup(raster) for camera in cameras]
    args.out.mkdir(parents=True,exist_ok=False)
    image_dir=args.out/'images'
    mask_dir=args.out/'masks'
    image_dir.mkdir()
    mask_dir.mkdir()
    calibration=dict(stream_order=list(streams),coordinates='Camera-pair frame in metres. Origin at midpoint, +X left-to-right, +Y up, -Z forward.',
                     pair_from_device=pair_from_device.tolist(),baseline_mm=float(np.linalg.norm(cameras[1].origin-cameras[0].origin)*1000),
                     angle_between_optical_axes_deg=float(np.degrees(np.arccos(np.clip(cameras[0].transform[:3,2]@cameras[1].transform[:3,2],-1,1)))),
                     source_calibration=args.calibration.name,source_sha256=digest(args.calibration),
                     cameras={name:dict(source_calibration=source[name]['calibration'],
                                       pair_from_eye=camera.transform.tolist(),validation=detail)
                              for name,camera,(_,detail) in zip(streams,cameras,combined)})
    (args.out/'rig.json').write_text(json.dumps(calibration,indent=2)+'\n')
    entries=[]
    for number in range(args.count):
        random=np.random.default_rng(np.random.SeedSequence([args.seed,number]))
        # The shared renderer uses index%9 to choose empty scenes.
        scene_index=0 if args.negative_every and number%args.negative_every==0 else 1
        vertices,faces,parts,shade,hands=scene(random,assets,triangles,scene_index)
        backdrop=assets[random.integers(len(assets))]['background']
        yaw=random.uniform(-np.pi,np.pi)
        gain=random.uniform(.85,1.5)
        entry=dict(id=f'{number:05d}',views={},
                   hands={SIDES[h['side']]:dict(asset=h['asset'],keypoints_pair_m=h['joints'].tolist()) for h in hands})
        for stream,side,camera,(lookup,_) in zip(streams,SIDES,cameras,combined):
            profile=(gain*random.uniform(.96,1.04),random.uniform(-8,13),random.uniform(.3,.85))
            image,mask=raster.image(camera,lookup,vertices,faces,parts,shade,backdrop,yaw,profile,random)
            mask=np.where(mask==3,0,mask).astype(np.uint8)
            name=f'{number:05d}-{side}.png'
            for directory,pixels in ((image_dir,image),(mask_dir,mask)):
                if not cv2.imwrite(str(directory/name),pixels,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                    raise OSError(f'Cannot save {directory/name}')
            visible=visible_boxes(mask)
            for hand in hands:
                key=SIDES[hand['side']]
                if key in visible:
                    pixels,depth=camera.project(hand['joints'])
                    visible[key]['keypoints_xy']=pixels.tolist()
                    visible[key]['keypoints_in_front']=(depth>0).tolist()
            entry['views'][stream]=dict(image=f'images/{name}',mask=f'masks/{name}',visible_hands=visible)
        entries.append(entry)
    manifest=dict(schema=1,seed=args.seed,source_sha256={p.name:digest(p) for p in args.source},
                  topology_sha256=digest(args.topology),asset_ids=[a['id'] for a in assets],
                  face_size=args.face_size,rig='rig.json',stream_order=list(streams),records=entries)
    (args.out/'dataset.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
    print(json.dumps(dict(pairs=len(entries),views=len(entries)*2,assets=len(assets),
                          hands_visible=sum(len(v['visible_hands']) for e in entries for v in e['views'].values()),
                          baseline_mm=calibration['baseline_mm'],cam_angle_deg=calibration['angle_between_optical_axes_deg'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit",type=Path,help="Override the installed calibration toolkit")
    for name in ('calibration','topology','library','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source',type=Path,nargs='+',required=True)
    parser.add_argument('--count',type=int,default=60)
    parser.add_argument('--asset-limit',type=int,default=100)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--negative-every',type=int,default=9)
    parser.add_argument('--face-size',type=int,default=640)
    run(parser.parse_args())
