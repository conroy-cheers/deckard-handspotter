# Handspotter release validation

## Workstation workflow

The public Nix smoke check generated temporary take-format video, exported reviewed boxes, and rendered three procedural groups on Chromium SwiftShader. It used the shipped mixed-data preparation command, verified view order and per-hand supervision, rejected a split leak, trained for one float epoch and one QAT epoch, resumed from the saved state, and exported the same integer artifact from the folded checkpoint. ONNX Runtime checked the raw and decoded ports. The tests use generated fixtures with no released dataset.

A separate ROCm reference run on Radeon RX 9070 used 192 procedural groups and 64 Frame groups with Terra-generated labels for training, 32 procedural groups for validation, and 32 procedural plus 48 Frame groups for test. It trained for 12 float epochs and two QAT epochs with batch size 12 and a 35% target recorded share. This run validates the workflow at pilot scale. The test recording was kept out of this scratch run's training set. Its labels were also Terra-generated. The recorded exports do not establish human review. Its relationship to older training runs for the deployed checkpoint is unknown.

| ONNX Runtime on the 48 recorded test groups | Centre hits / labeled slots | IoU ≥ 0.5 hits | False positives / reviewed negatives |
|---|---:|---:|---:|
| Scratch workflow pilot | 26 / 202 | 1 | 48 / 181 |
| Deployed release checkpoint | 195 / 202 | 106 | 27 / 181 |

Both rows use the same native-image preprocessing and a 0.2 confidence threshold. The small procedural pilot's recorded-frame result supports keeping the deployed checkpoint for the release. The QNN builder generated an HTP context from the pilot's integer ONNX. The pilot checkpoint and the recordings remain local. [Final checkpoint and historical comparison](handspotter-model-card.md)

## Reproduce the checks

```sh
nix build --no-link .#handspotter.tests.smoke .#frame-hand-labeler-tools
```

Use the [training walkthrough](handspotter-training.md) for the workstation workflow and the [take format](frame-takes.md) for recorded input.

The pilot drew 256 groups with replacement per epoch, targeting 35% recorded groups. It used seed 899, BF16 autocast, fused AdamW with weight decay 0.006, per-view brightness multipliers from 0.9 to 1.1, and cosine schedules starting at 0.001 for the float phase and 0.00012 for QAT. Float epoch 10 and QAT epoch 14 were selected on the synthetic validation set. On the 80 mixed test groups, centre hits were 145/319 in float, 54/319 after QAT and 57/319 in integer ONNX. [Pilot dataset recipe](pilot/dataset.toml) · [Saved pilot settings](pilot/train.toml) · [Aggregate test results](pilot/holdout.json)
