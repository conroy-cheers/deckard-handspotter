"""Bake rigged MPFB characters and fitted MHCLO garments using headless Blender."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from head_pose import in_reference_scene
from humanoid_geometry import (
    SIDES, arm_rotation, body_basis, forearm_rotation, hand_record, pair_points,
    palm_frame, reach_targets, scene_record, wrist_rotation, write_bank,
)


def read_hands(bones, suffix):
    points = [np.asarray(bones[f"wrist.{suffix}"].head)]
    for digit in range(1,6):
        chain = [bones[f"finger{digit}-{part}.{suffix}"] for part in range(1,4)]
        points.extend(np.asarray(bone.head) for bone in chain)
        points.append(np.asarray(chain[-1].tail))
    return np.array(points)


def turn(bone, delta):
    transform = bone.matrix.copy()
    new = Matrix(delta.tolist()).to_4x4()@transform
    new.translation = transform.translation
    bone.matrix = new


def apply_fingers(bones, suffix, source):
    if source is None:
        return
    donor = np.asarray(source["keypoints_pair_m"])
    local_to_rig = palm_frame(read_hands(bones,suffix))@palm_frame(donor).T
    for digit in range(1,6):
        for part in (2,3):
            bone = bones[f"finger{digit}-{part}.{suffix}"]
            i = 1+4*(digit-1)+part-1
            current = bone.tail-bone.head
            desired = Vector(local_to_rig@(donor[i+1]-donor[i]))
            rotation = current.rotation_difference(desired)
            if rotation.angle>1.15:
                rotation = Quaternion(rotation.axis,1.15)
            turn(bone,np.asarray(rotation.to_matrix()))
            bpy.context.view_layer.update()


def label_skin(body):
    body.data.materials.clear()
    for side in (*SIDES,"body"):
        body.data.materials.append(bpy.data.materials.new(f"part-{side}"))
    by_group = {group.index:i for i,suffix in enumerate(("L","R"))
                for group in body.vertex_groups
                if group.name.endswith("."+suffix) and
                group.name.startswith(("wrist.","metacarpal","finger"))}
    weights = np.zeros((len(body.data.vertices),2))
    for vertex in body.data.vertices:
        for group in vertex.groups:
            if group.group in by_group:
                weights[vertex.index,by_group[group.group]] += group.weight
    for polygon in body.data.polygons:
        weight = weights[list(polygon.vertices)].mean(axis=0)
        polygon.material_index = int(weight.argmax()) if weight.max()>=.5 else 2


def surface(obj, rig, basis, eyes, skin):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    to_rig = np.asarray(rig.matrix_world.inverted()@evaluated.matrix_world)
    mesh = evaluated.to_mesh()
    try:
        vertices = np.empty((len(mesh.vertices),3),np.float32)
        mesh.vertices.foreach_get("co",vertices.ravel())
        vertices = vertices@to_rig[:3,:3].T+to_rig[:3,3]
        mesh.calc_loop_triangles()
        faces = np.empty((len(mesh.loop_triangles),3),np.int32)
        mesh.loop_triangles.foreach_get("vertices",faces.ravel())
        labels = np.full(len(faces),3,np.uint8)
        if skin:
            parts = np.empty(len(faces),np.int32)
            mesh.loop_triangles.foreach_get("material_index",parts)
            labels = (parts+1).astype(np.uint8)
        return pair_points(vertices,basis,eyes).astype(np.float32),faces,labels
    finally:
        evaluated.to_mesh_clear()


def build_character(args, asset, variant):
    from bl_ext.user_default.mpfb.services import HumanService, TargetService

    shape = TargetService.get_default_macro_info_dict()
    rng = np.random.default_rng(np.random.SeedSequence([args.seed,variant]))
    shape.update(gender=float(rng.uniform(.15,.85)), height=float(rng.uniform(.44,.90)),
                 weight=float(rng.uniform(.34,.64)), muscle=float(rng.uniform(.40,.66)))
    body = HumanService.create_human(macro_detail_dict=shape)
    rig = HumanService.add_builtin_rig(body,"default")
    clothes = HumanService.add_mhclo_asset(str(asset.resolve()),body,
                        material_type="NONE",subdiv_levels=0)
    if not any(mod.type=="ARMATURE" and mod.object==rig for mod in clothes.modifiers):
        raise ValueError(f"Garment {asset.stem} was not bound to the character")
    for obj in (body,clothes):
        for mod in obj.modifiers:
            if mod.type=="ARMATURE":
                mod.use_deform_preserve_volume = True
    label_skin(body)
    bones = rig.pose.bones
    basis = body_basis(np.asarray(bones["upperarm01.L"].head),
                      np.asarray(bones["upperarm01.R"].head), np.array([0.,0.,1.]))
    return body,rig,clothes,basis


def pose(rig, basis, source, rng):
    bones = rig.pose.bones
    for bone in bones:
        bone.matrix_basis = Matrix.Identity(4)
    for name in ("spine02","spine01","clavicle.L","clavicle.R"):
        bone = bones[name]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = rng.normal(0,np.deg2rad([3,4,4]))
    bpy.context.view_layer.update()
    eyes = np.array([bones[name].head for name in ("eye.L","eye.R")]).mean(axis=0)
    result = {}
    for side,suffix in zip(SIDES,("L","R"),strict=True):
        shoulder,elbow,wrist = (bones[f"{name}.{suffix}"] for name in
                               ("upperarm01","lowerarm01","wrist"))
        s,e,w = (np.asarray(bone.head) for bone in (shoulder,elbow,wrist))
        target_elbow,target_wrist = reach_targets(source,side,s,e,w,basis,eyes)
        palm = palm_frame(read_hands(bones,suffix))
        turn(shoulder,arm_rotation(s,e,target_elbow,target_wrist,palm,-basis[:,2]))
        bpy.context.view_layer.update()
        palm = palm_frame(read_hands(bones,suffix))
        turn(elbow,forearm_rotation(np.asarray(shoulder.head),np.asarray(elbow.head),
                                   np.asarray(wrist.head),target_wrist,palm[:,2]))
        bpy.context.view_layer.update()
        donor = source["hands"].get(side)
        apply_fingers(bones,suffix,donor)
        turn(wrist,wrist_rotation(read_hands(bones,suffix),np.asarray(elbow.head),donor,basis))
        bpy.context.view_layer.update()
        s,e = pair_points([shoulder.head,elbow.head],basis,eyes)
        points = pair_points(read_hands(bones,suffix),basis,eyes)
        result[side] = hand_record(source,"mpfb-body",side,points,s,e)
    return eyes, result


def bake(args):
    source = json.loads(args.source.read_text())["records"][args.start:args.start+args.count]
    if len(source)!=args.count or args.start<0 or min(args.count,args.builds)<1:
        raise ValueError("Invalid scene range or character count")
    source = [in_reference_scene(row) for row in source]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    count = args.builds*len(args.garment)
    characters = []
    for i in range(count):
        asset = args.garment[i%len(args.garment)]
        characters.append(build_character(args,asset,i))
    args.out.mkdir(parents=True)
    records = []
    for i,row in enumerate(source):
        index = (i+args.start)%len(characters)
        body,rig,clothes,basis = characters[index]
        asset = args.garment[index%len(args.garment)]
        rng = np.random.default_rng(np.random.SeedSequence([args.seed,i+args.start]))
        eyes,hands = pose(rig,basis,row,rng)
        data = {}
        for key,obj in (("skin",body),("cloth",clothes)):
            vertices,faces,labels = surface(obj,rig,basis,eyes,key=="skin")
            if not np.isfinite(vertices).all() or not len(faces):
                raise ValueError(f"Failed to export the deformed {key} mesh")
            data.update({f"{key}_vertices":vertices,f"{key}_faces":faces,f"{key}_labels":labels})
        if not all((data["skin_labels"]==label).sum()>100 for label in (1,2)):
            raise ValueError("The posed MPFB hand regions were lost")
        appearance = dict(row["appearance"], garment=asset.stem,
                          cloth_roughness=.85, pattern=0, cuff=-1)
        records.append(dict(scene_record(row,"mpfb-body",hands,garment=asset.stem,
                                         build=index), appearance=appearance))
        np.savez_compressed(args.out/f"{i:05d}.npz",**data)
        if (i+1)%8==0 or i+1==args.count:
            print(f"Baked {i+1}/{args.count} MPFB bodies",flush=True)
    checksums = {}
    for asset in args.garment:
        for file in sorted(asset.parent.iterdir()):
            if file.is_file():
                with file.open("rb") as data:
                    checksums[f"{asset.stem}/{file.name}"] = hashlib.file_digest(data,"sha256").hexdigest()
    write_bank(args.out,args.source,dict(library="MPFB",garment_sha256=checksums),records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--garment",type=Path,action="append",required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--count",type=int,required=True)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--builds",type=int,default=3,help="How many body shapes per garment")
    bake(parser.parse_args(sys.argv[sys.argv.index("--")+1:]))
