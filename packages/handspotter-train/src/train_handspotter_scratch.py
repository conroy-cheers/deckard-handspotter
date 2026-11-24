"""Train Handspotter on simulated Frame inputs and optional warped HOT3D clips."""
import argparse
import copy
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F

from hot3d_training import FrontCorpus
from handspotter_data import detector_metrics, load_synthetic, render_image_path, synthetic_identities, targets, write_json
from handspotter_shards import digest, load_split, split_shards
from handspotter_model import dsp_fast_path_layers
from handspotter_torch import Handspotter, exports
from handspotter_metrics import evaluate_export


def verify_rig(directory, rig_hash):
    metadata = json.loads((directory/"rig.json").read_text())
    if metadata["source_sha256"]!=rig_hash or any(
        camera["source_calibration"]["projection_model_type"]!="kb4"
        for camera in metadata["cameras"].values()
    ):
        raise ValueError(f"Dataset {directory} does not use the declared Frame rig")
    records = json.loads((directory/"dataset.json").read_text())["records"]
    return records


def load_dataset(directories, rig_hash):
    sets = []
    assets = set()
    for directory in directories:
        records = verify_rig(directory, rig_hash)
        values = load_synthetic(directory, "validation", 0, 0)
        assets.update(synthetic_identities(records, values[0]))
        sets.append(values)
    return tuple(torch.from_numpy(np.concatenate(items)) for items in zip(*sets,strict=True)),assets


def detection_loss(center, size, logits, box, valid):
    truth,diameter,present = targets(box,valid)
    present_f = present.to(center)
    number = present_f.sum().clamp_min(1)
    center_error = F.smooth_l1_loss(center/.1,truth/.1,reduction="none",beta=.25).sum(dim=-1)
    size_error = F.smooth_l1_loss(size/.1,diameter/.1,reduction="none",beta=.25)
    scores = F.binary_cross_entropy_with_logits(logits,present_f,reduction="none")
    return ((center_error+.45*size_error)*present_f).sum()/number+2*(scores*valid).sum()/valid.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, data, device, batch, threshold):
    model.eval()
    images,boxes,valid = data
    chunks = [[],[],[]]
    for samples in images.split(batch):
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            c,s,p = model(samples.to(device))
        for dest,value in zip(chunks,(c*1280,s.clamp_min(0)*640,p.sigmoid()),strict=True):
            dest.append(value.float().cpu().numpy())
    return detector_metrics(*(np.concatenate(chunk) for chunk in chunks),boxes.numpy(),valid.numpy(),threshold=threshold)


def selection_score(scores, args):
    false_rate = scores["empty_slot_false_positives"] / max(1, scores["reviewed_negative_slots"])
    return (scores["center_recall"] + args.selection_iou_weight * scores["iou50"] / scores["labels"]
            - args.selection_false_positive_weight * false_rate)


def training_batch(synthetic, hot3d, indices):
    if hot3d is None:
        return tuple(field[indices] for field in synthetic)
    count = len(synthetic[0])
    real_rows = indices >= count
    synth_rows = ~real_rows
    result = tuple(field.new_empty((len(indices), *field.shape[1:])) for field in synthetic)
    if synth_rows.any():
        for output, field in zip(result, synthetic, strict=True):
            output[synth_rows] = field[indices[synth_rows]]
    if real_rows.any():
        values = hot3d.batch(indices[real_rows].numpy() - count)
        for output, field in zip(result, values, strict=True):
            output[real_rows] = torch.from_numpy(field)
    return result


def train_phase(model, train, dev, device, args, epochs, epoch_offset, rate, history, out, hot3d=None, sampling=None):
    optimizer = torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=args.weight_decay,fused=device.type=="cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=rate*args.min_rate_ratio)
    phase = "qat" if model.qat else "float"
    checkpoint = out/f"{phase}-resume.pt"
    best, state = -math.inf,None
    completed = 0
    if args.resume and checkpoint.is_file():
        saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history[:] = saved["history"]
        completed, best, state = saved["completed"], saved["best"], saved["best_model"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["accelerator_rng"])
        print(f"Resuming {phase} training after epoch {epoch_offset+completed}",flush=True)
    elif model.qat or args.initial_checkpoint:
        scores = evaluate(model,dev,device,args.batch,args.threshold)
        best,state = selection_score(scores,args),copy.deepcopy(model.state_dict())
        history.append(dict(epoch=epoch_offset,phase="qat-initial" if model.qat else "float-initial",
                            validation=scores,score=best))
    for step in range(completed+1,epochs+1):
        epoch = epoch_offset+step
        model.train()
        losses = []
        group_count = len(train[0]) + (len(hot3d) if hot3d is not None else 0)
        order = (torch.randperm(group_count) if sampling is None else
                 torch.multinomial(sampling, group_count, replacement=True))
        batches = order.split(args.batch)
        for index,subset in enumerate(batches):
            images,boxes,valid = (field.to(device) for field in training_batch(train,hot3d,subset))
            if args.brightness:
                gain = 1 + args.brightness * (2 * torch.rand((len(images), 4, 1, 1), device=device) - 1)
                images = (images.float() * gain).round().clamp(0, 255).to(torch.uint8)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                loss = detection_loss(*model(images),boxes,valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            losses.append(loss.detach())
            if index%100==0:
                print(f"Epoch {epoch} batch {index}/{len(batches)} loss {float(loss.detach()):.4f}",flush=True)
        scores = evaluate(model,dev,device,args.batch,args.threshold)
        mean_loss = float(torch.stack(losses).mean())
        if not math.isfinite(mean_loss):
            raise ValueError("Non-finite training loss")
        quality = selection_score(scores,args)
        summary = dict(epoch=epoch,phase=phase,
                       loss=mean_loss,validation=scores,score=quality)
        history.append(summary)
        if quality>best:
            best,state = quality,copy.deepcopy(model.state_dict())
            torch.save(state,out/f"{phase}-best.pt")
        scheduler.step()
        if args.resume:
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),rng=torch.get_rng_state(),
                            accelerator_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None,
                            completed=step,best=best,best_model=state,history=history),temporary)
            temporary.replace(checkpoint)
        write_json(out/"history.json",history)
        print(f"Epoch {epoch} validation recall {scores['center_recall']:.1%}, "
              f"IoU50 {scores['iou50']/scores['labels']:.1%}, score {quality:.4f}",flush=True)
    model.load_state_dict(state)


def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True,exist_ok=True)
    device = torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA or ROCm device")
    rig_hash = digest(args.rig) if args.rig else None
    hot3d = FrontCorpus(args.hot3d_training, rig_hash) if args.hot3d_training else None
    if args.dataset:
        train, dev, test = (
            tuple(torch.from_numpy(array) for array in load_split(args.dataset, split))
            for split in ("training", "validation", "test")
        )
        train_assets, dev_assets, test_assets = (), (), ()
    else:
        train, train_assets = load_dataset(args.training, rig_hash)
        dev, dev_assets = load_dataset(args.validation, rig_hash)
        test, test_assets = load_dataset(args.test, rig_hash)
        if train_assets & (dev_assets | test_assets) or dev_assets & test_assets:
            raise ValueError("Pose asset leakage between training, validation and test")
    for name, data in zip(("training", "validation", "test"), (train, dev, test), strict=True):
        if not ((data[1][..., 2:] > data[1][..., :2]).all(dim=-1) & data[2]).any():
            raise ValueError(f"Need at least one supervised hand box in the {name} split")
    sampling = None
    if args.recorded_fraction is not None:
        sources = split_shards(args.dataset, "training")
        kinds = [item["kind"] for item in sources for _ in range(item["groups"])]
        recorded = kinds.count("recorded")
        synthetic = kinds.count("synthetic")
        if not recorded or not synthetic:
            raise ValueError("Source weighting requires both recorded and synthetic training groups")
        sampling = torch.tensor([
            args.recorded_fraction / recorded if kind == "recorded"
            else (1 - args.recorded_fraction) / synthetic for kind in kinds
        ])
    model = Handspotter()
    if args.initial_checkpoint:
        state = torch.load(args.initial_checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(state)
    model.to(device=device,memory_format=torch.channels_last)
    history = []
    real_groups = len(hot3d) if hot3d is not None else 0
    report = dict(seed=args.seed,groups=[len(train[0])+real_groups,len(dev[0]),len(test[0])],
                  pose_assets=[sum(kind == "pose" for kind, _ in ids) for ids in (train_assets, dev_assets, test_assets)],rig_sha256=rig_hash,
                  random_initialization=args.initial_checkpoint is None,
                  initial_checkpoint_sha256=digest(args.initial_checkpoint) if args.initial_checkpoint else None,
                  backend=str(device),float_epochs=args.epochs,qat_epochs=args.qat_epochs,
                  weight_decay=args.weight_decay,min_rate_ratio=args.min_rate_ratio,
                  qat_rate_ratio=args.qat_rate_ratio,brightness=args.brightness,
                  recorded_fraction=args.recorded_fraction,threshold=args.threshold,
                  selection_iou_weight=args.selection_iou_weight,
                  selection_false_positive_weight=args.selection_false_positive_weight,
                  rate=args.rate,threads=args.threads,architecture=1,torch=str(torch.__version__),
                  onnx=onnx.__version__,onnxruntime=ort.__version__,
                  dataset_sha256=digest(args.dataset / "dataset.json") if args.dataset else None,
                  batch=args.batch,trainable_parameters=sum(p.numel() for p in model.parameters()))
    if hot3d is not None:
        report.update(synthetic_training_groups=len(train[0]), hot3d_training_groups=real_groups,
                      hot3d_revision=hot3d.revision)
    run_config = args.out/"run.json"
    if args.resume:
        report.update(training=[str(path.resolve()) for path in (args.training or [])],
                      validation=[str(path.resolve()) for path in (args.validation or [])],
                      test=[str(path.resolve()) for path in (args.test or [])],
                      hot3d_manifest_sha256=digest(args.hot3d_training/"training.json") if hot3d else None)
        if not args.dataset:
            report["render_inputs"] = [
                {"path": str(directory.resolve()), "images": digest(render_image_path(directory)),
                 "labels": digest(directory / "dataset.json"), "rig": digest(directory / "rig.json")}
                for directory in (*args.training, *args.validation, *args.test)
            ]
        if not run_config.is_file() and any(args.out.glob("*-resume.pt")):
            raise ValueError("Missing run configuration for saved checkpoints")
        if run_config.is_file() and json.loads(run_config.read_text()) != report:
            raise ValueError(f"Training configuration changed for {args.out}")
    write_json(run_config,report)
    print(json.dumps(report),flush=True)
    train_phase(model,train,dev,device,args,args.epochs,0,args.rate,history,args.out,hot3d,sampling)
    float_test = evaluate(model,test,device,args.batch,args.threshold)
    model.fold_normalization()
    if args.qat_epochs:
        train_phase(model,train,dev,device,args,args.qat_epochs,args.epochs,args.rate*args.qat_rate_ratio,history,args.out,hot3d,sampling)
    torch.save(model.state_dict(),args.out/"folded.pt")
    qat_test = evaluate(model,test,device,args.batch,args.threshold)
    raw,production = exports(model)
    onnx.save(raw,args.out/"handspotter-frame-integer.onnx")
    onnx.save(production,args.out/"handspotter-frame-dequantized.onnx")
    test_arrays = tuple(field.numpy() for field in test)
    integer_test = evaluate_export(production, test_arrays, args.threads, args.threshold)
    write_json(args.out/"holdout.json",dict(float=float_test,qat=qat_test,integer=integer_test))
    write_json(args.out/"dsp.json",dict(fast_path=dsp_fast_path_layers(raw)))
    print(f"Full integer ONNX held-out center recall {integer_test['center_recall']:.1%}",flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training",type=Path,nargs="+")
    parser.add_argument("--hot3d-training",type=Path,help="Add the full warped HOT3D front-camera corpus")
    parser.add_argument("--validation",type=Path,nargs="+")
    parser.add_argument("--test",type=Path,nargs="+")
    parser.add_argument("--initial-checkpoint",type=Path,help="Optional float Handspotter checkpoint for a controlled fine-tune")
    parser.add_argument("--resume",action="store_true",help="Save and resume training at completed epochs")
    parser.add_argument("--config", type=Path, help="TOML defaults, with paths relative to the config")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rig", type=Path)
    parser.add_argument("--dataset", type=Path, help="Prepared mixed dataset directory")
    parser.add_argument("--seed",type=int,default=899)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="PyTorch device (default: %(default)s, selected from available hardware)")
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--qat-epochs",type=int,default=2)
    parser.add_argument("--batch",type=int,default=24)
    parser.add_argument("--threads",type=int,default=5)
    parser.add_argument("--rate",type=float,default=.001)
    parser.add_argument("--threshold", type=float, default=.2)
    parser.add_argument("--selection-iou-weight", type=float, default=.20)
    parser.add_argument("--selection-false-positive-weight", type=float, default=.10)
    parser.add_argument("--weight-decay", type=float, default=.006)
    parser.add_argument("--min-rate-ratio", type=float, default=.08)
    parser.add_argument("--qat-rate-ratio", type=float, default=.12)
    parser.add_argument("--recorded-fraction", type=float,
                        help="Target recorded share in weighted sampling with replacement")
    parser.add_argument("--brightness", type=float, default=0,
                        help="Maximum fractional per-view brightness augmentation")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config = tomllib.loads(preliminary.config.read_text())
        path_fields = {"dataset", "rig", "initial_checkpoint", "out", "hot3d_training"}
        split_fields = {"training", "validation", "test"}
        options = {action.dest: action for action in parser._actions}
        for key, value in config.items():
            if key not in options or key in ("config", "help"):
                parser.error(f"Unknown config option {key!r}")
            if key in path_fields:
                config[key] = (preliminary.config.parent / value).resolve()
            elif key in split_fields:
                config[key] = [(preliminary.config.parent / item).resolve() for item in value]
            elif type(value) not in (int, float, str, bool):
                parser.error(f"Invalid config value for {key!r}")
            elif isinstance(options[key].default, bool) and type(value) is not bool:
                parser.error(f"Expected a boolean config value for {key!r}")
            elif options[key].type in (int, float) and type(value) not in (
                    (int,) if options[key].type is int else (int, float)):
                parser.error(f"Expected a numeric config value for {key!r}")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if args.out is None:
        parser.error("Supply --out or set out in the config")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        parser.error("Output is nonempty. Use --resume to continue the same run.")
    if (not all(map(math.isfinite, (args.rate, args.weight_decay, args.min_rate_ratio,
                                    args.qat_rate_ratio, args.brightness, args.threshold,
                                    args.selection_iou_weight, args.selection_false_positive_weight)))
            or not 0 <= args.brightness <= 1 or args.weight_decay < 0
            or not 0 <= args.threshold <= 1 or args.selection_iou_weight < 0
            or args.selection_false_positive_weight < 0
            or not 0 <= args.min_rate_ratio <= 1 or args.qat_rate_ratio <= 0):
        parser.error("Invalid augmentation or optimizer parameters")
    if args.recorded_fraction is not None and (
            not 0 < args.recorded_fraction < 1 or not args.dataset or args.hot3d_training):
        parser.error("--recorded-fraction needs a mixed prepared dataset and a fraction between zero and one")
    if args.dataset and any((args.training, args.validation, args.test)):
        parser.error("Use either --dataset or separate synthetic shards")
    if not args.dataset and not all((args.training, args.validation, args.test, args.rig)):
        parser.error("Supply --dataset or --rig with all three synthetic splits")
    if args.hot3d_training and not args.rig:
        parser.error("--hot3d-training requires --rig")
    if min(args.epochs,args.batch,args.threads)<1 or args.rate<=0 or args.qat_epochs<0:
        parser.error("Use positive epochs, batch, thread count and learning rate")
    return args


if __name__ == "__main__":
    try:
        main(parse_args())
    except (OSError, ValueError) as error:
        raise SystemExit(f"handspotter-train: {error}") from None
