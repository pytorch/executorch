# MobileSAM Prompt Segmentation Example Application

This end-to-end example shows how to use the Arm Ethos-U backend in
ExecuTorch for transformer-based prompt segmentation. MobileSAM predicts a
binary mask for fixed positive point prompts rather than semantic class IDs.
The host debug flow validates quantization by comparing FP32 and quantized
masks, with an optional binary reference mask when one is available.

It covers:

- Loading the MobileSAM `vit_t` checkpoint.
- Freezing one or more positive point prompts into the exported graph.
- Applying post-training quantization with the Ethos-U quantizer.
- Lowering the quantized model to an Ethos-U85-256 ExecuTorch program.
- Producing validation and debugging artifacts such as masks, overlays,
  mismatch heatmaps, metrics, and delegation summaries.
- Building a bare-metal Corstone-320 runtime app and running it on FVP.

The default export uses a reduced `448x448` image input and returns one
low-resolution `[1, 1, 112, 112]` mask-logit tensor. The example prepares the
official MobileSAM GitHub source at a pinned revision in an external checkout
and applies a small configurable-input patch there. Neither the MobileSAM
source nor checkpoint is redistributed in ExecuTorch.

The export uses int8 activations and int8 weights globally, and A16W8
quantization for TinyViT attention modules. This keeps the transformer
attention numerically stable while still producing one Ethos-U delegate.

The exported graph intentionally uses `multimask_output=False` and leaves
mask thresholding outside the model. SAM-style candidate-mask selection can be
numerically sensitive after export and quantization, so this example keeps the
target graph focused on the fixed-prompt image encoder and mask decoder.

## Layout

- `model_export/prepare_mobilesam.py` - Prepares the pinned external MobileSAM
  checkout and applies the configurable-input patch.
- `model_export/README.md` - Model loading, quantization, lowering,
  validation, and debug artifact generation.
- `runtime/README.md` - Bare-metal runtime build, image header generation, and
  Corstone-320 FVP execution.
- `runtime/visualize_fvp_output.py` - Decodes the target mask dump, creates an
  overlay, and compares FVP output with the host quantized mask.
