## Person detection example on Cortex-M

This example demonstrates training, exporting and deployment of a person detection model based on [microYOLO: Towards Single-Shot Object Detection on Microcontrollers](https://arxiv.org/pdf/2408.15865) [1] in a small demo application.

Before you begin, install Executorch and relevant Cortex-M dependencies according to the [documentation](https://docs.pytorch.org/executorch/stable/backends/arm-cortex-m/arm-cortex-m-overview.html). Additional pip packages required for the demo are listed in `requirements.txt`.


## Model and training
The model architecure is based on the approach laid out in [1], with a few modifications. The class channel C is removed since there is
only one class predicted, and a finer 7x7 grid with one single box is used rather than the suggested 5x5 grid and two boxes. Using the `SxSx(C + B*5)` formula his reduces the number of parameters from `5x5x(1+2*5)=275` to `7x7x(1*5)=245`, and it simplifies the training since a cell detection is binary true/false, rather than assigned to a particular box. When tested over the validation set, this representation leads to a decrease around 3% in the ability to match the true lables, which is minor compared to
the final accuracy loss of the constrained model.

The model backbone goes through a pretraining phase on Caltech256 [2] in `training/pretrain.py`, and the full training is then done one Open Images V7 Person detection [3] in `training/training.py` with custom image transforms and loss function. To support the pruning applied in the paper, a mask is increasingly applied during the last 100 epochs of training. The model is then compacted using the `training/pruning.py` script which halves the number of channels of the model as it removes masked columns.

To test the capabilities of the trained model, `training/test_model.py` by default will output one random image from Open Images Person detection annotated by the model, or if you have access to a webcamera you may use `training/test_model.py --webcam` to test it on completely live data. Since the model is based on a work-in-progress paper under heavy constraints the accuracy is not expected to be top quality (~27mAP@0.5 reported on the COCO dataset in the original paper, ~20mAP@0.5 seen over Open Images V7), but simpler samples provided by e.g. a webcam with one-three people present are generally well detected from experience.

## Export
The export folder contains the Executorch AOT pt2 quantization and lowering using the Cortex-M backend. Addtionally the general QuantizeInputs/Outputs passes are applied to make the graph run directly on the int8 data provided by the camera, rather than the standard float inputs/outputs and quantization/ dequantization operators.

A testing script similar the one used when training is also provided to easily compare the accuracy between the original and lowered model.

## Deployment
TODO

## Citations and Licensing

[1] Deutel, Mutschler, and Teich (2024)
microYOLO: Towards Single-Shot Object Detection on Microcontrollers
arXiv:2408.15865
Licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).
The implementation code has been adapted for ExecuTorch and is licensed under the BSD-style
license found in the LICENSE file in the root directory of this source tree.

[2] Griffin, G. Holub, AD. Perona, P.
The Caltech 256.
Caltech Technical Report.
Dataset licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)

[3] Kuznetsova, A., Rom, H., Alldrin, N., et al. (2020).
The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale.
International Journal of Computer Vision.
Open Images V7 annotations are licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).
Individual images are listed as having a [Creative Commons Attribution 2.0 Generic](https://creativecommons.org/licenses/by/2.0/) license and remain subject to their respective attribution and other applicable rights.
Users are responsible for verifying and preserving image-level attribution when redistributing dataset content.
