# EfficientSAM Model Export

This example demonstrates how to export the [EfficientSAM](https://github.com/yformer/EfficientSAM) model to Core ML and XNNPACK using ExecuTorch.

# Instructions

## 1. Setup

Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup#) to set up ExecuTorch.

## 2. Exports

### Exporting to Core ML

Make sure to install the [required dependencies](https://pytorch.org/executorch/main/build-run-coreml.html#setting-up-your-developer-environment) for Core ML export.

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

# Licensing

The code in the `efficient_sam_core` directory is licensed under the [Apache License 2.0](./efficient_sam_core/LICENSE.txt).
