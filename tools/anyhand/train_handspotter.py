"""Train and verify staged Handspotter integer checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from torch.nn import functional as F

from frame_camera import digest
from handspotter_cameras import stereo_streams
from stereo_images import resize_gray
from handspotter_data import LIMIT, QUAD_STREAMS, SIDES, detector_metrics, load_synthetic, targets, write_json
from hot3d_clip import open_clip
from handspotter_qat import FEATURE, DetectorTail, prefix_session, verify_export
from handspotter_model import check_interface


STAGES = ("002.5", "005", "007.5")
TEST_STAGES = ("000", *STAGES, "010", "025")


def load_real(clips,selection,split):
    images,boxes,valid = [],[],[]
    for item in selection:
        if item["split"]!=split:
            continue
        path = clips/f"clip-{item['clip']:06d}.tar"
        if digest(path)!=item["sha256"]:
            raise ValueError(f"HOT3D archive changed at {path}")
        with open_clip(path) as clip:
            for frame in item["frames"]:
                info = clip.json(f"{frame:06d}.info.json")
                if (info["participant_id"] != item["participant"] or ("device" in item and
                        info["device"].casefold() != item["device"].casefold())):
                    raise ValueError(f"Unexpected subject or camera at {path} frame {frame}")
                hands = clip.json(f"{frame:06d}.hands.json")
                image = np.zeros((4,128,160),np.uint8)
                label = np.zeros((4,2,4),np.float32)
                supervised = np.zeros((4,2),bool)
                cameras = clip.json(f"{frame:06d}.cameras.json")
                for view, stream in enumerate(stereo_streams(cameras), 1):
                    pixels = clip.pixels(frame, stream)
                    cal = cameras[stream]["calibration"]
                    h, w = pixels.shape
                    if (w, h) != (cal["image_width"], cal["image_height"]):
                        raise ValueError(f"Image dimensions differ from calibration for {stream}")
                    image[view] = resize_gray(pixels, 160, 128)
                    scale = LIMIT / [w, h, w, h]
                    for hand,side in enumerate(SIDES):
                        row = hands.get(side,{})
                        bounds = row.get("boxes_amodal",{}).get(stream)
                        if bounds is None:
                            supervised[view,hand] = True
                        elif row["visibilities_modeled"].get(stream,0)>.1:
                            bound = np.clip(bounds * scale, 0, LIMIT)
                            if np.all(bound[2:]>bound[:2]):
                                supervised[view,hand] = True
                                label[view,hand] = bound
                images.append(image)
                boxes.append(label)
                valid.append(supervised)
    return tuple(map(np.asarray,(images,boxes,valid)))


def cache(args):
    cv2.setNumThreads(1)
    model = onnx.load(args.model)
    encoder = prefix_session(model)
    if args.synthetic:
        images,boxes,valid = load_synthetic(args.synthetic,args.split,args.dropout,args.seed)
    else:
        selection = json.loads(args.selection.read_text())
        parts = [{r["participant"] for r in selection if r["split"]==s} for s in ("training","validation","test")]
        if sum(len(p) for p in parts)!=len(set.union(*parts)):
            raise ValueError("HOT3D participants overlap splits")
        images,boxes,valid = load_real(args.clips,selection,args.split)
    if not len(images) or images.dtype!=np.uint8:
        raise ValueError("No valid Handspotter byte input in this split")
    def one(image):
        return encoder.run([FEATURE],{"images":image[None]})[0][0]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        features = np.stack(list(pool.map(one,images)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.out.with_suffix(".part")
    with temporary.open("wb") as file:
        np.savez(file,features=features,boxes=boxes,valid=valid,model_sha256=digest(args.model),
                 samples=images[:8])
    temporary.replace(args.out)
    print(f"{args.out.name}: {len(images)} inputs, {int((images[:,0:4:3]==0).all(axis=(1,2,3)).sum())} blank-side groups",flush=True)


def load_cache(path, model_sha):
    with np.load(path) as file:
        if file["model_sha256"].item()!=model_sha:
            raise ValueError(f"Stale features at {path}")
        return tuple(torch.from_numpy(file[name]) for name in ("features","boxes","valid"))


def loss_values(model, raw, box, valid, teacher, anchor):
    center,size,logit = model.decode(raw)
    target_center,target_size,present = targets(box,valid)
    classification = F.binary_cross_entropy_with_logits(logit[valid],present[valid].float(),reduction="sum")
    number_present = present.sum().clamp(min=1)
    err = F.smooth_l1_loss(center[present]/.1,target_center[present]/.1,beta=.35,reduction="sum")
    extent = F.smooth_l1_loss(size[present]/.1,target_size[present]/.1,beta=.35,reduction="sum")
    prior_center,_,prior_logit = model.decode(teacher)
    preserve = present&anchor[:,None,None]&(prior_logit>0)
    drift = F.smooth_l1_loss(center[preserve]/.1,prior_center[preserve]/.1,beta=.35,reduction="sum")
    return ((2*err+.45*extent)/number_present
            +.45*classification/valid.sum().clamp(min=1)+.15*drift/preserve.sum().clamp(min=1))


@torch.no_grad()
def evaluate(model,data,common=None):
    encoded,truth,valid = data
    center_out,size_out,conf_out = [],[],[]
    for features in encoded.split(32):
        raw = model(features.float())
        center,size,_ = model.decode(raw)
        center_out.append(center.numpy()*1280)
        size_out.append(size.numpy()*640)
        conf_out.append(model.sigmoid[raw[2].long()].numpy()/255.)
    center,radius,confidence = map(np.concatenate,(center_out,size_out,conf_out))
    return detector_metrics(center,radius,confidence,truth.numpy(),valid.numpy(),common)


def shuffled_batches(size,batch,steps,rng):
    # Every geometry sees all its examples before sampling the next permutation.
    chunks = [rng.permutation(size) for _ in range((batch*steps+size-1)//size)]
    return np.concatenate(chunks)[:batch*steps].reshape(steps,batch)


def quality(scores):
    anchor,shifted,real = scores
    return .25*anchor["center_recall"]+.50*shifted["center_recall"]+.25*real["center_recall"]


def train(args):
    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled=False
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    source = onnx.load(args.model)
    source_sha = digest(args.model)
    model = DetectorTail(source)
    teacher = DetectorTail(source).eval()
    prefix = prefix_session(source)
    load = lambda name: load_cache(args.features/f"{name}.npz",source_sha)
    training_anchor,training_real = load("training-000"),load("training-hot3d")
    validation_anchor,validation_real = load("validation-000"),load("validation-hot3d")
    with np.load(args.features/"validation-000.npz") as f:
        verify_images = f["samples"][:4].copy()
    with np.load(args.features/"validation-hot3d.npz") as f:
        verify_images = np.concatenate((verify_images,f["samples"][:4]))
    verify_export(model,source,prefix,verify_images)
    decoded = check_interface(onnx.load(args.decoded_model),decoded=True) if args.decoded_model else None
    geometry = STAGES[-1] if args.direct else args.direct_geometry
    stages = list((geometry,)*args.rounds if geometry else (args.schedule or STAGES)[:args.rounds])
    report = dict(model_sha256=source_sha,feature=FEATURE,threshold=.2,schedule=stages,
                  size_convention="diameter",training_rounds=[])
    original = copy.deepcopy(model.state_dict())
    for index,stage in enumerate(stages,1):
        if index<args.start_round:
            summary = json.loads((args.out/f"round-{index}.json").read_text())
            if summary["round"]!=index or summary["geometry"]!=stage:
                raise ValueError(f"Round {index} on disk does not match the requested schedule")
            model.load_state_dict(torch.load(args.out/f"round-{index}-state.pt",weights_only=True))
            report["training_rounds"].append(summary)
            continue
        shifted = load(f"training-{stage}")
        validation = (validation_anchor,load(f"validation-{stage}"),validation_real)
        train_sets = (training_anchor,shifted,training_real)
        initial = [evaluate(model,data) for data in validation]
        best, selected = quality(initial),0
        best_weights = copy.deepcopy(model.state_dict())
        history = [dict(epoch=0,validation=initial,score=best)]
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.02)
        batch = (12,12,6)
        steps = max((len(data[0])+size-1)//size for data,size in zip(train_sets,batch))
        print(f"Round {index} geometry {stage}, starts at {best:.4f}",flush=True)
        for epoch in range(1,args.epochs+1):
            sequences = [shuffled_batches(len(dataset[0]),size,steps,rng) for dataset,size in zip(train_sets,batch)]
            flags = torch.tensor([True]*batch[0]+[False]*batch[1]+[True]*batch[2])
            losses = []
            for indices in zip(*sequences):
                feature,boxes,valid = (torch.cat([data[field][rows] for data,rows in zip(train_sets,indices)]) for field in range(3))
                feature = feature.float()
                with torch.no_grad():
                    baseline = teacher(feature)
                optimizer.zero_grad()
                loss = loss_values(model,model(feature),boxes,valid,baseline,flags)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scores = [evaluate(model,data) for data in validation]
            metric = quality(scores)
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=scores,score=metric))
            if metric>best:
                best,selected = metric,epoch
                best_weights = copy.deepcopy(model.state_dict())
            print(index,epoch,[round(s["center_recall"],4) for s in scores],round(metric,4),flush=True)
        model.load_state_dict(best_weights)
        artifact = model.export()
        with np.load(args.features/f"validation-{stage}.npz") as f:
            round_verify_images = np.concatenate((verify_images,f["samples"][:4]))
        verify_export(model,artifact,prefix,round_verify_images)
        onnx.save(artifact,args.out/f"round-{index}-integer.onnx")
        if decoded is not None:
            onnx.save(model.export(decoded),args.out/f"round-{index}-dequantized.onnx")
        torch.save(best_weights,args.out/f"round-{index}-state.pt")
        summary = dict(round=index,geometry=stage,selected_epoch=selected,history=history,
                       training_groups=[len(data[0]) for data in train_sets],batch_groups=list(batch),
                       seed=args.seed,learning_rate=args.learning_rate,verified_integer_inputs=len(round_verify_images))
        write_json(args.out/f"round-{index}.json",summary)
        report["training_rounds"].append(summary)
    write_json(args.out/"training.json",report)

    if args.evaluate:
        heldout = {name:load(f"test-{name}") for name in (*args.test_geometries,"hot3d")}
        common = np.logical_and.reduce([targets(*heldout[name][1:])[2].numpy() for name in args.test_geometries])
        checkpoints = {f"round-{index}":args.out/f"round-{index}-state.pt" for index in range(1,args.rounds+1)}
        checkpoints.update((name,Path(path)) for name,path in args.reference)
        outcomes = {}
        for name,path in [("round-0",None),*checkpoints.items()]:
            state = torch.load(path,weights_only=True) if path else original
            model.load_state_dict(state)
            outcomes[name] = {
                cohort:evaluate(model,data,common if cohort!="hot3d" else None)
                for cohort,data in heldout.items()
            }
            recall = ", ".join(f"{cohort} {metrics['center_recall']:.1%}"
                               for cohort,metrics in outcomes[name].items())
            print(f"{name}: {recall}",flush=True)
        write_json(args.out/"holdout.json",dict(model_sha256=source_sha,schedule=stages,outcomes=outcomes))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(required=True)
    prep=sub.add_parser("cache")
    for name in ("model","out"):
        prep.add_argument(f"--{name}",type=Path,required=True)
    prep.add_argument("--split",choices=("training","validation","test"),required=True)
    source=prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic",type=Path)
    source.add_argument("--selection",type=Path)
    prep.add_argument("--clips",type=Path)
    prep.add_argument("--workers",type=int,default=2)
    prep.add_argument("--dropout",type=float,default=.2)
    prep.add_argument("--seed",type=int,default=44)
    prep.set_defaults(run=cache)
    fit=sub.add_parser("train")
    for name in ("model","features","out"):
        fit.add_argument(f"--{name}",type=Path,required=True)
    fit.add_argument("--decoded-model",type=Path)
    fit.add_argument("--epochs",type=int,default=8)
    fit.add_argument("--threads",type=int,default=4)
    fit.add_argument("--learning-rate",type=float,default=.06)
    fit.add_argument("--seed",type=int,default=501)
    fit.add_argument("--rounds",type=int,help="Number of rounds to run from the schedule")
    fit.add_argument("--start-round",type=int,default=1)
    schedule=fit.add_mutually_exclusive_group()
    schedule.add_argument("--direct",action="store_true",help="Use the 7.5%% geometry in each round as a curriculum control")
    schedule.add_argument("--direct-geometry",help="Geometry cache suffix to use in all rounds, e.g. 100")
    schedule.add_argument("--schedule",nargs="+",help="Geometry cache suffix for each round, in order")
    fit.add_argument("--test-geometries",nargs="+",default=TEST_STAGES)
    fit.add_argument("--reference",action="append",nargs=2,default=[],metavar=("NAME","STATE"))
    fit.add_argument("--evaluate",action="store_true")
    fit.set_defaults(run=train)
    args=parser.parse_args()
    if args.run==cache and args.selection and not args.clips:
        parser.error("--selection requires --clips")
    if args.run==train:
        configured_rounds = len(args.schedule or STAGES)
        if args.rounds is None:
            args.rounds = configured_rounds
        if args.rounds<1 or (not (args.direct or args.direct_geometry) and args.rounds>configured_rounds):
            parser.error("--rounds must fit the selected schedule")
        if not 1<=args.start_round<=args.rounds+1:
            parser.error("--start-round must be between 1 and --rounds plus 1")
        names = [name for name,_ in args.reference]
        if len(set(names))!=len(names) or any(name.startswith("round-") for name in names):
            parser.error("Reference names must be unique and may not start with round-")
        if len(set(args.test_geometries))!=len(args.test_geometries) or "hot3d" in args.test_geometries:
            parser.error("Test geometry names must be unique; HOT3D is included automatically")
    args.run(args)
