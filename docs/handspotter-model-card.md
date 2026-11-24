# Handspotter Frame checkpoint

The repository includes a trained [integer ONNX checkpoint](../artifacts/handspotter-frame-integer.onnx) for the Frame's Hexagon DSP. Pin the repository revision to select the same weights for deployment. The checkpoint satisfies the export contract, and `conv0`–`conv5` and `conv12`–`conv15` qualify for the staged U16 path.

Handspotter detects the wearer's anatomical left and right hands in four synchronized calibrated Steam Frame tracking-camera views. A separate stereo pose stage runs after detection. The native gaze path is separate from Handspotter. The model consumes uint8 `[1,4,128,160]` and emits uint8 hand centres, sizes and confidence scores, plus `conv11` features that a downstream temporal model can optionally consume. The graph structure and fixed-point conversion are documented in the [source model definition](../lib/model/handspotter_model.py).

This checkpoint was selected after on-headset testing. On an existing 300-group native replay, the recorded comparison had 1,107 labeled slots and 1,273 reviewed empty slots. The release candidate hit 771 labeled centres and produced 270 false positives in reviewed empty slots. The saved baseline hit 636 centres and produced 285 false positives. This comparison characterizes one recorded cohort. It does not establish performance for new subjects, all lighting conditions, fast motion or overlapping hands.

The checkpoint is a blend of trained models. Its training data combined AnyHand Studio renders, video derived from HOT3D and the author's own Frame recordings. Its complete training receipt still needs reconstruction from the original experiments. The public scratch workflow produces independently trainable weights with the same interface. [Training workflow](handspotter-training.md)

## License

The checkpoint weights are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0). Its training labels include HOT3D hand annotations, which the [HOT3D license](https://projectaria.com/datasets/hot3D/license) releases under CC BY-NC-SA. Models you train with this repository's code carry the terms of the data you train them on.

The project source code is licensed under MIT. Third-party dependencies and assets retain their own terms. The checkpoint is the only model or data asset in the repository.
