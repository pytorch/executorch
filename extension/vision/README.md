# ExecuTorch Vision Extension

Image preprocessing and postprocessing utilities for on-device inference with ExecuTorch.

## Overview

This module provides `ImagePreprocessor` and `ImagePostprocessor` classes that generate `ExportedProgram` objects ready to be lowered to any ExecuTorch backend (CoreML, MPS, XNNPACK, etc.).

### Key Features

- **SDR Support**: 8-bit sRGB content (photos, standard video)
- **HDR Support**: 10/12-bit PQ (HDR10, Dolby Vision) and HLG (broadcast HDR)
- **Color Gamut Conversion**: BT.2020 ↔ BT.709 wide color gamut support
- **Standard Normalizations**: ImageNet, [0,1], [-1,1] scaling
- **Color Layout**: RGB ↔ BGR conversion, grayscale
- **Precision Control**: fp16/fp32 output dtype selection

## Quick Start

```python
from executorch.extension.vision import ImagePreprocessor, ImagePostprocessor

# Simple [0, 255] → [0, 1] preprocessing
preprocessor = ImagePreprocessor.from_scale_0_1(
    shape=(1, 3, 480, 640),
    input_dtype=torch.float16,
)

# ImageNet normalization
preprocessor = ImagePreprocessor.from_imagenet(
    shape=(1, 3, 224, 224),
    input_dtype=torch.float16,
)

# HDR10 → linear BT.709 (for HDR video processing)
preprocessor = ImagePreprocessor.from_hdr10(
    shape=(1, 3, 1080, 1920),
    input_dtype=torch.float32,   # fp32 recommended for PQ precision
    output_dtype=torch.float16,
)

# Lower to your backend
from executorch.exir import to_edge_transform_and_lower
program = to_edge_transform_and_lower(preprocessor, partitioner=[YourPartitioner()])
```

## ImagePreprocessor

Converts input images to normalized tensors for model inference.

### Factory Methods

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| `from_scale_0_1()` | [0, 255] | [0, 1] | Simple normalization |
| `from_scale_neg1_1()` | [0, 255] | [-1, 1] | Zero-centered models |
| `from_imagenet()` | [0, 255] | ImageNet normalized | Classification models |
| `from_sdr()` | 8-bit sRGB | Linear or normalized | SDR with gamma correction |
| `from_hdr10()` | 10-bit PQ BT.2020 | Linear BT.709 | HDR10/Dolby Vision |
| `from_hlg()` | 10-bit HLG BT.2020 | Linear BT.709 | Broadcast HDR |

### Processing Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────┐
│ 1. Color Layout         │  BGR → RGB (if needed)
├─────────────────────────┤
│ 2. Normalize to [0,1]   │  Divide by max_value (255, 1023, 4095)
├─────────────────────────┤
│ 3. Inverse Transfer     │  PQ/HLG/sRGB → Linear
├─────────────────────────┤
│ 4. Gamut Conversion     │  BT.2020 → BT.709 (if needed)
├─────────────────────────┤
│ 5. Output Transfer      │  Linear → sRGB (if needed)
├─────────────────────────┤
│ 6. Bias & Scale         │  (x + bias) * scale
├─────────────────────────┤
│ 7. Output Layout        │  RGB → Grayscale/BGR (if needed)
└─────────────────────────┘
    │
    ▼
Normalized Tensor
```

## ImagePostprocessor

Converts model output tensors back to displayable images. Performs the inverse operations of `ImagePreprocessor`.

### Factory Methods

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| `from_scale_0_1()` | [0, 1] | [0, 255] | Simple denormalization |
| `from_scale_neg1_1()` | [-1, 1] | [0, 255] | Zero-centered models |
| `from_imagenet()` | ImageNet normalized | [0, 255] | Classification models |
| `from_linear_to_srgb()` | Linear | 8-bit sRGB | SDR output |
| `from_linear_to_hdr10()` | Linear BT.709 | 10-bit PQ BT.2020 | HDR10 output |
| `from_linear_to_hlg()` | Linear BT.709 | 10-bit HLG BT.2020 | Broadcast HDR output |

## HDR Processing

### Precision Considerations

⚠️ **Important**: PQ (HDR10/Dolby Vision) transfer functions require **fp32 compute precision** for accurate results. The PQ EOTF contains an exponent of m2=78.84, which causes significant precision loss when computed in fp16.

| Transfer Function | Recommended Precision | Notes |
|-------------------|----------------------|-------|
| **PQ (HDR10)** | fp32 input | m2=78.84 exponent loses precision in fp16 |
| **HLG** | fp16 OK | Piecewise sqrt/log is fp16-friendly |
| **sRGB** | fp16 OK | Simple x^2.2 power function |

### HDR10 (PQ / Dolby Vision)

```python
# Preprocessing: HDR10 input → linear for model
# ⚠️ Use fp32 input_dtype for accurate PQ decoding
preprocessor = ImagePreprocessor.from_hdr10(
    shape=(1, 3, 1080, 1920),
    input_dtype=torch.float32,   # IMPORTANT: fp32 required for PQ precision
    output_dtype=torch.float16,  # Output can be fp16 for model inference
)

# Postprocessing: model output → HDR10 display
# ⚠️ Use fp32 output_dtype for accurate PQ encoding
postprocessor = ImagePostprocessor.from_linear_to_hdr10(
    shape=(1, 3, 1080, 1920),
    input_dtype=torch.float16,
    output_dtype=torch.float32,  # IMPORTANT: fp32 required for PQ precision
)
```

### HLG (Broadcast HDR)

HLG uses a piecewise sqrt/log function that is more numerically stable in fp16.

```python
# Preprocessing: HLG input → linear for model
preprocessor = ImagePreprocessor.from_hlg(
    shape=(1, 3, 1080, 1920),
    input_dtype=torch.float16,   # fp16 OK for HLG
    output_dtype=torch.float16,
)

# Postprocessing: model output → HLG display
postprocessor = ImagePostprocessor.from_linear_to_hlg(
    shape=(1, 3, 1080, 1920),
    input_dtype=torch.float16,
    output_dtype=torch.float16,  # fp16 OK for HLG
)
```

## Platform Integration Notes

The following operations should be done **outside** the model using platform-native APIs for best performance:

| Operation | iOS | Android |
|-----------|-----|---------|
| uint8 ↔ float | vDSP (~1ms for 512×512) | RenderScript |
| YUV → RGB | vImage | YuvImage |
| Resize/Crop | vImage, Metal | Bitmap APIs |

The preprocessor/postprocessor handles the **numerically intensive** operations (transfer functions, gamut conversion, normalization) that benefit from acceleration.

## Transfer Functions

### Supported Functions

| Function | Standard | Use Case |
|----------|----------|----------|
| **sRGB** | IEC 61966-2-1 | Standard displays, web |
| **PQ** | SMPTE ST.2084 | HDR10, Dolby Vision |
| **HLG** | BT.2100 | Broadcast HDR (BBC/NHK) |
| **Linear** | - | ML models, compositing |

### Implementation Details

- **PQ (Perceptual Quantizer)**: Uses the ST.2084 EOTF with constants m1=0.1593, m2=78.84, c1=0.8359, c2=18.85, c3=18.69. The large m2 exponent causes precision loss in fp16 - use fp32 for accurate results.

- **HLG (Hybrid Log-Gamma)**: Piecewise function with sqrt for low values and log for high values. More fp16-friendly than PQ.

- **sRGB**: Uses the x^2.2 / x^(1/2.2) approximation rather than the piecewise sRGB transfer function.

## Color Gamut

### Supported Gamuts

| Gamut | Standard | Coverage |
|-------|----------|----------|
| **BT.709** | Rec. 709 / sRGB | Standard SDR |
| **BT.2020** | Rec. 2020 | Wide color gamut (HDR) |

### Conversion Matrices

The module includes accurate 3×3 matrices for BT.709 ↔ BT.2020 conversion, validated against the `colour-science` reference library.

## Testing

Tests compare implementations against the `colour-science` library which provides reference implementations of ITU/SMPTE standards.

```bash
# Run tests
python -m unittest extension.vision.test.test_image_processing -v

# Or directly
python extension/vision/test/test_image_processing.py -v
```

### Test Coverage

- Transfer function accuracy vs colour-science reference
- Gamut conversion matrix validation
- Full E2E pipeline tests for all factory methods
- fp16 and fp32 precision validation
- Roundtrip tests (forward → inverse)

## API Reference

### ImagePreprocessor

```python
class ImagePreprocessor(torch.nn.Module):
    """
    Args:
        bit_depth: Input bit depth (8, 10, or 12). Default 8.
        input_transfer: Transfer function of input (SRGB, PQ, HLG, LINEAR).
        output_transfer: Desired transfer function of output.
        input_gamut: Color gamut of input (BT709, BT2020).
        output_gamut: Desired color gamut of output.
        input_color: Color layout of input (RGB, BGR).
        output_color: Desired color layout (RGB, BGR, GRAYSCALE).
        channel_bias: Per-channel bias [R, G, B].
        channel_scale: Per-channel scale [R, G, B].
        preset: Preset name ("scale_0_1", "scale_neg1_1", "imagenet").
        output_dtype: Output dtype (torch.float16 or torch.float32).
    """
```

### ImagePostprocessor

```python
class ImagePostprocessor(torch.nn.Module):
    """
    Args:
        bit_depth: Output bit depth (8, 10, or 12). Default 8.
        input_transfer: Transfer function of input (LINEAR, SRGB).
        output_transfer: Desired transfer function of output (SRGB, PQ, HLG).
        input_gamut: Color gamut of input (BT709, BT2020).
        output_gamut: Desired color gamut of output.
        input_color: Color layout of input (RGB, BGR, GRAYSCALE).
        output_color: Desired color layout (RGB, BGR).
        preset: Preset name for inverse normalization.
        output_dtype: Output dtype (torch.float16 or torch.float32).
    """
```
