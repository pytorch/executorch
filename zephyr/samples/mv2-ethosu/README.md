# MobileNetV2 Image Classification with Ethos-U NPU

This sample demonstrates running a quantized MobileNetV2 image classification
model on the Arm Ethos-U NPU using ExecuTorch within a Zephyr RTOS application.

The model classifies a static RGB test input tensor with shape `[1, 3, 224, 224]`
(NCHW) into one of 1000 ImageNet classes and prints the top-5 predictions.

## Prerequisites

- Zephyr SDK with ExecuTorch module enabled
- Python 3.10+ with ExecuTorch, torchvision, and ethos-u-vela installed
- A board with Arm Ethos-U NPU (e.g., Corstone-300 FVP, Alif E7/E8 DevKit)

## Export the model

Export a quantized INT8 MobileNetV2 model with Ethos-U delegation:

```bash
python -m modules.lib.executorch.backends.arm.scripts.aot_arm_compiler \
    --model_name=mv2_untrained \
    --quantize \
    --delegate \
    --target=ethos-u55-128 \
    --output=mv2_ethosu.pte
```

For boards with Ethos-U55-256 (e.g., Alif E8 HP core), use `--target=ethos-u55-256`.

## Build

### Corstone-300 FVP

```bash
west build -b mps3/corstone300/fvp \
    modules/lib/executorch/zephyr/samples/mv2-ethosu \
    -t run -- \
    -DET_PTE_FILE_PATH=mv2_ethosu.pte
```

### Alif Ensemble E8 DevKit

```bash
west build -b alif_e8_dk/ae822fa0e5597xx0/rtss_hp \
    -S ethos-u55-enable \
    modules/lib/executorch/zephyr/samples/mv2-ethosu -- \
    -DET_PTE_FILE_PATH=mv2_ethosu.pte
```

## Expected output

```
========================================
ExecuTorch MobileNetV2 Classification Demo
========================================

Ethos-U backend registered successfully
Model loaded, has 1 methods
Inference completed in <N> ms

--- Classification Results ---
Top-5 predictions:
  [1] class <id>: <score>
  [2] class <id>: <score>
  ...

MobileNetV2 Demo Complete
Inference time: <N> ms
========================================
```

When using `mv2_untrained`, the output class IDs will be arbitrary since the
model has no trained weights. Use `mv2` (requires torchvision pretrained
weights) for meaningful predictions.

## Memory requirements

The default configuration allocates 1.5 MB each for the method and temporary
allocator pools. These defaults are sufficient for a fully NPU-delegated INT8
MobileNetV2. Adjust `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` and
`CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE` in `prj.conf`, in board-specific
`*.conf` files (for example, `boards/<board>.conf`), or via Zephyr's
`OVERLAY_CONFIG` mechanism for different model configurations.
