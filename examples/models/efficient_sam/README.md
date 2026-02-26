# EfficientSAM Model Export

This example demonstrates how to export the [EfficientSAM](https://github.com/yformer/EfficientSAM) model to Core ML and XNNPACK using ExecuTorch.

# Instructions

## 1. Setup

Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup#) to set up ExecuTorch.

## 2. Exports

### Exporting to Core ML

Make sure to install the [required dependencies](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html#development-requirements) for Core ML export.

To export the model to Core ML, run the following command:

```bash
cd executorch
python -m examples.apple.coreml.scripts.export -m efficient_sam
```

### Exporting to XNNPACK

To export the model to XNNPACK, run the following command:

```bash
cd executorch
python -m examples.xnnpack.aot_compiler -m efficient_sam
```

# Performance

Tests were conducted on an Apple M1 Pro chip using the instructions for building and running Executorch with [Core ML](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html#runtime-integration) and [XNNPACK](https://docs.pytorch.org/executorch/main/tutorial-xnnpack-delegate-lowering.html#running-the-xnnpack-model-with-cmake) backends.

| Backend Configuration  | Average Inference Time (seconds) |
| ---------------------- | -------------------------------- |
| Core ML (CPU, GPU, NE) | 34.8                             |
| Core ML (CPU, GPU)     | 34.7                             |
| Core ML (CPU, NE)      | 26.4                             |
| Core ML (CPU)          | 22.8                             |
| XNNPACK                | 4.1                              |

All models were tested with `float32` precision.

# Licensing

The code in the `efficient_sam_core` directory is licensed under the [Apache License 2.0](efficient_sam_core/LICENSE.txt).
