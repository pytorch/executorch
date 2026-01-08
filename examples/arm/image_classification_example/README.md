# Image Classification Example Application

This end-to-end example shows how to use the Arm backend in ExecuTorch across both ahead-of-time (AoT) and runtime flows. It covers
this by providing examples of:

- Scripts to fine-tune a DeiT-Tiny model on the Oxford-IIIT Pet dataset, quantize it, and export an Ethos-U–ready ExecuTorch program.
- A simple bare-metal image-classification app for Corstone-320 (Ethos-U85-256) that embeds the exported program and a sample image.
- Running the app on the Corstone-320 Fixed Virtual Platform (FVP).

## Layout

The example is divided into two sections:

- `model_export/README.md` — Covers fine-tuning a model for a new usecase, quantization to INT8, lowering to Ethos-U via ExecuTorch and `.pte` generation.
- `runtime/README.md` — Covers building the bare-metal app, generating headers from the `.pte` and image, and running on the FVP.

In addition, this example uses `../executor_runner/` for various utilities (linker scripts, memory allocators, and the PTE-to-header converter).
