# Espressif Quantization

The Espressif backend delegates only int8 operators, so a model must be quantized before lowering. `EspNNQuantizer` is a PT2E quantizer that annotates a model with exactly the scheme the ESP-NN kernels execute. Operators it leaves in float run on the CPU.

## Supported Quantization Schemes

The Espressif delegate supports the following quantization scheme:

- **8-bit static quantization.**
  - Activations are int8, per-tensor affine, with scales computed during calibration.
  - Weights are int8, per-output-channel symmetric. Per-tensor weights are available with `EspNNQuantizer(per_channel=False)`.
  - Biases are int32.

`EspNNQuantizer` builds on `XNNPACKQuantizer` and adds the following ESP-NN specifics:

- **Fused activations.** A ReLU or ReLU6 that directly follows a `conv2d`, `linear` or `add` is fused into that operator. The clamp then applies to the operator's requantized output, as ESP-NN's activation range arguments implement it, and ReLU6 doesn't waste most of the int8 range on a full-range scale.
- **Shared quantization.** Pooling, resize, padding and data movement operators, and eval-mode `dropout`, reuse their input's quantization parameters, so no requantization is needed between them.
- **Softmax.** The softmax output uses a fixed scale of 1/256 and zero point -128, which the delegate's softmax table requires.
- **Splits.** `split`, `split_with_sizes`, `chunk` and `unbind` are rewritten to slices before annotation, so that every piece can carry a quantization annotation.

## Post-Training Quantization using the PT2E Flow

To quantize a model after training:

1. Export the model in eval mode with `torch.export.export`.
2. Insert observers with `prepare_pt2e` and an `EspNNQuantizer`.
3. Run calibration data through the prepared model.
4. Convert it with `convert_pt2e`, export the result again, and lower it with `to_espnn_edge`.

```python
import torch
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = model.eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

prepared = prepare_pt2e(
    torch.export.export(model, example_inputs).module(), EspNNQuantizer()
)
with torch.no_grad():
    for images in calibration_batches:   # a few hundred real samples
        prepared(images)
quantized = convert_pt2e(prepared)

edge, report = to_espnn_edge(torch.export.export(quantized, example_inputs))
```

Calibrate on representative data from your dataset. Gaussian noise or a handful of samples gives scales that don't match real activations, and accuracy drops sharply.

## Quantization-Aware Training

Networks with hardswish and squeeze-and-excitation blocks, such as MobileNetV3 and EfficientNet, lose much more accuracy with post-training quantization alone. Quantization-aware training (QAT) recovers most of it. Pass `is_qat=True` and prepare with `prepare_qat_pt2e`:

```python
from torchao.quantization.pt2e import move_exported_model_to_eval, move_exported_model_to_train
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

prepared = prepare_qat_pt2e(
    torch.export.export(model, example_inputs).module(),
    EspNNQuantizer(is_qat=True),
)
move_exported_model_to_train(prepared)
for images, labels in train_batches:
    loss = loss_fn(prepared(images), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
move_exported_model_to_eval(prepared)
quantized = convert_pt2e(prepared)
```

[`examples/espressif/camera_stream/export_model.py`](https://github.com/pytorch/executorch/blob/main/examples/espressif/camera_stream/export_model.py) implements QAT by distillation from the float model, so it needs no labels. For MobileNetV3-Small at 224×224, on 200 Imagenette validation images (1000-way top-1):

| Quantization | Correct |
|---|---|
| Float model | 134 |
| PTQ, 1024 calibration images | 19 |
| QAT, 3 epochs, learning rate 1e-4 | 64 |
| QAT, 5 epochs, learning rate 1e-3 | 150 to 156 (three runs) |

Each of the three runs retrained from scratch: two on ExecuTorch 1.6.0 and one on 1.4.1. The final QAT result beats the float model because the distillation fine-tunes on Imagenette's 10 classes; it isn't a general ImageNet result. The accuracy of the int8 program is measured with the backend's host emulator, which reproduces the delegate's arithmetic exactly. On an ESP32-S3, the program's outputs for sample validation images were identical to the emulator's.

## Per-Module Configuration

`EspNNQuantizer` accepts the same per-module configuration as `XNNPACKQuantizer`. A module-name config takes precedence over a module-type config, which takes precedence over the global config:

```python
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)

quantizer = EspNNQuantizer().set_module_name(
    "classifier", get_symmetric_quantization_config(is_per_channel=False)
)
```

The fused-activation, lookup-table and shared-quantization annotations follow the configuration that governs each node.
