# Image Classification Example Application (VGF)

This example shows how to export a DEiT-Tiny model for the Arm VGF backend and
run it on host using the generic `executor_runner` binary. It is a host-only
workflow; a device-specific VGF runtime application is out of scope here.

## Layout

- `model_export/README.md` — Fine-tuning a model, quantization to INT8, lowering
  to VGF via ExecuTorch, and `.pte` generation.
- `runtime/README.md` — Running the VGF `.pte` on host using `executor_runner`.

Use `examples/arm/image_classification_example_ethos_u` for the Ethos-U
bare-metal flow.
