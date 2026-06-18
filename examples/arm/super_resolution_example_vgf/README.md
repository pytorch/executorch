# Swin2SR Super-Resolution Example Application (VGF)

This example shows how to export a Swin2SR image super-resolution model for the
Arm VGF backend and run it on host using the generic `executor_runner` binary.
It is a host-only workflow; a device-specific VGF runtime application is out of
scope here.

## Layout

- `model_export/prepare_demo_assets.py` — Creates a deterministic text-heavy
  demo input plus small LR/HR calibration and evaluation sets from a repo-local
  screenshot.
- `model_export/README.md` — Dataset-backed FP/INT8 export, PTQ
  calibration and evaluation, and `.pte` generation.
- `runtime/README.md` — Running the exported `.pte` on host using
  `executor_runner` and converting the output tensor back into an image.

Use `examples/arm/image_classification_example_vgf` for the image
classification flow.
