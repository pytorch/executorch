# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Image preprocessing and postprocessing utilities for on-device inference.

This module provides:
- ImagePreprocessor: Convert input images to normalized tensors for model inference
- ImagePostprocessor: Convert model output tensors back to displayable images

Common operations handled:
- Color space conversion (RGB ↔ BGR, grayscale)
- Per-channel bias and scale normalization
- HDR support (PQ, HLG transfer functions)
- Wide color gamut (BT.2020 ↔ BT.709 conversion)
- Multiple bit depths (8, 10, 12-bit)

NOTE: The following should be done OUTSIDE the model using platform-native APIs:
- uint8 ↔ float conversion (use vDSP on iOS, ~1ms for 512x512)
- YUV → RGB conversion (use vImage on iOS)
- Resize/crop (use vImage or Metal on iOS)

Usage Examples:
    from executorch.extension.vision import ImagePreprocessor

    # Get an ExportedProgram for [0, 255] → [0, 1] preprocessing
    ep = ImagePreprocessor.from_scale_0_1(
        shape=(1, 3, 480, 640),
        input_dtype=torch.float16,
    )

    # For HDR10 with float32 precision (recommended for accurate PQ)
    ep = ImagePreprocessor.from_hdr10(
        shape=(1, 3, 1080, 1920),
        input_dtype=torch.float32,  # Use float32 for HDR precision
        output_dtype=torch.float16,
    )

    # Lower to any backend
    from executorch.exir import to_edge_transform_and_lower
    program = to_edge_transform_and_lower(ep, partitioner=[YourPartitioner()])
"""

import warnings
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.export


# =============================================================================
# Enums
# =============================================================================


class ColorLayout(Enum):
    """Color layout options matching CoreML's color_layout."""

    RGB = "RGB"
    BGR = "BGR"
    GRAYSCALE = "GRAYSCALE"


class TransferFunction(Enum):
    """Transfer function (gamma curve) options."""

    SRGB = "srgb"  # Standard sRGB gamma (~2.2)
    LINEAR = "linear"  # Linear (no gamma)
    PQ = "pq"  # Perceptual Quantizer (HDR10, Dolby Vision)
    HLG = "hlg"  # Hybrid Log-Gamma (broadcast HDR)


class ColorGamut(Enum):
    """Color gamut options."""

    BT709 = "bt709"  # Standard sRGB/Rec.709 (SDR)
    BT2020 = "bt2020"  # Wide color gamut (HDR)


# =============================================================================
# Shared Constants
# =============================================================================

# BT.2020 to BT.709 color matrix (3x3)
BT2020_TO_BT709 = torch.tensor(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ]
)

# BT.709 to BT.2020 color matrix (inverse)
BT709_TO_BT2020 = torch.tensor(
    [
        [0.6274, 0.3293, 0.0433],
        [0.0691, 0.9195, 0.0114],
        [0.0164, 0.0880, 0.8956],
    ]
)

# Luminance weights for RGB to grayscale conversion
LUMINANCE_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


# =============================================================================
# Shared Transfer Functions
# =============================================================================


def apply_srgb_gamma(x: torch.Tensor) -> torch.Tensor:
    """Convert linear to sRGB gamma. Approximation: x^(1/2.2)"""
    return torch.pow(x.clamp(min=1e-6), 1.0 / 2.2)


def apply_srgb_inverse(x: torch.Tensor) -> torch.Tensor:
    """Convert sRGB gamma to linear. Approximation: x^2.2"""
    return torch.pow(x.clamp(min=1e-6), 2.2)


def apply_pq_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Convert PQ (Perceptual Quantizer) to linear.
    PQ is used in HDR10 and Dolby Vision.
    """
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    x = x.clamp(min=1e-6, max=1.0)
    x_m2 = torch.pow(x, 1.0 / m2)
    numerator = (x_m2 - c1).clamp(min=0.0)
    denominator = (c2 - c3 * x_m2).clamp(min=1e-6)
    return torch.pow(numerator / denominator, 1.0 / m1)


def apply_pq_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Convert linear to PQ (Perceptual Quantizer).
    Inverse of apply_pq_inverse.
    """
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    x = x.clamp(min=1e-6, max=1.0)
    x_m1 = torch.pow(x, m1)
    numerator = c1 + c2 * x_m1
    denominator = 1.0 + c3 * x_m1
    return torch.pow(numerator / denominator, m2)


def apply_hlg_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Convert HLG (Hybrid Log-Gamma) to linear.
    HLG is used in broadcast HDR (BBC/NHK).
    """
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    x = x.clamp(min=1e-6, max=1.0)
    low = (x * x) / 3.0
    high = (torch.exp((x - c) / a) + b) / 12.0
    return torch.where(x <= 0.5, low, high)


def apply_hlg_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Convert linear to HLG (Hybrid Log-Gamma).
    Inverse of apply_hlg_inverse.
    """
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    x = x.clamp(min=1e-6, max=1.0)
    low = torch.sqrt(3.0 * x)
    high = a * torch.log(12.0 * x - b) + c
    return torch.where(x <= 1.0 / 12.0, low, high)


def apply_gamut_conversion(x: torch.Tensor, color_matrix: torch.Tensor) -> torch.Tensor:
    """Apply 3x3 color matrix for gamut conversion."""
    # [B, 3, H, W] -> [B, H, W, 3] for matmul
    x = x.permute(0, 2, 3, 1)
    x = torch.matmul(x, color_matrix.T)
    # [B, H, W, 3] -> [B, 3, H, W]
    x = x.permute(0, 3, 1, 2)
    return x.clamp(min=0.0, max=1.0)


# =============================================================================
# Shared Presets
# =============================================================================

# Standard normalization presets
# For preprocessor: bias and scale applied as output = (input + bias) * scale
# For postprocessor: inverse applied as output = (input / scale) - bias
PRESETS = {
    # Simple pass-through [0, 1] -> [0, 1]
    "none": {
        "bias": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    },
    # Scale [0, 1] to [0, 1] (identity, but explicit)
    "scale_0_1": {
        "bias": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    },
    # Zero-centered [-1, 1] from [0, 1] input
    "scale_neg1_1": {
        "bias": [-0.5, -0.5, -0.5],
        "scale": [2.0, 2.0, 2.0],
    },
    # ImageNet normalization from [0, 1] input
    # Formula: (x - mean) / std
    "imagenet": {
        "bias": [-0.485, -0.456, -0.406],
        "scale": [1 / 0.229, 1 / 0.224, 1 / 0.225],
    },
}


# =============================================================================
# ImagePreprocessor
# =============================================================================


class ImagePreprocessor(torch.nn.Module):
    """
    Comprehensive image preprocessing model - replacement for CoreML's ImageType.

    Handles all common preprocessing for image/video models:
    - SDR: 8-bit sRGB content (photos, standard video)
    - HDR: 10/12-bit PQ or HLG content (HDR10, Dolby Vision, broadcast HDR)
    - Color layout conversion (RGB ↔ BGR, grayscale)
    - Per-channel bias and scale normalization
    - Color gamut conversion (BT.2020 ↔ BT.709)

    Processing pipeline:
    1. Convert input color layout (BGR → RGB if needed)
    2. Normalize to [0, 1] based on bit depth
    3. Apply inverse transfer function (linearize): gamma/PQ/HLG → linear
    4. Apply color gamut conversion (BT.2020 → BT.709 if needed)
    5. Apply output transfer function (linear → gamma if needed)
    6. Apply per-channel bias and scale (normalization)
    7. Convert to output color layout (grayscale, BGR if needed)

    Args:
        bit_depth: Input bit depth (8, 10, or 12). Default 8 for SDR.
        input_transfer: Transfer function of input (SRGB, PQ, HLG, LINEAR).
        output_transfer: Desired transfer function of output (SRGB, LINEAR).
        input_gamut: Color gamut of input (BT709, BT2020). Default BT709.
        output_gamut: Desired color gamut of output (BT709, BT2020). Default BT709.
        input_color: Color layout of input (RGB, BGR). Default RGB.
        output_color: Desired color layout (RGB, BGR, GRAYSCALE). Default RGB.
        channel_bias: Per-channel bias [R, G, B] applied after all conversions.
        channel_scale: Per-channel scale [R, G, B] applied after bias.
        preset: Optional preset name that sets bias/scale.
        output_dtype: Output data type (torch.float16 or torch.float32).

    Input:
        float16 tensor [B, C, H, W] with values in [0, max_val] based on bit_depth

    Output:
        float16/float32 tensor [B, C', H, W] normalized and converted
    """

    def __init__(
        self,
        bit_depth: int = 8,
        input_transfer: TransferFunction = TransferFunction.LINEAR,
        output_transfer: TransferFunction = TransferFunction.LINEAR,
        input_gamut: ColorGamut = ColorGamut.BT709,
        output_gamut: ColorGamut = ColorGamut.BT709,
        input_color: ColorLayout = ColorLayout.RGB,
        output_color: ColorLayout = ColorLayout.RGB,
        channel_bias: Optional[List[float]] = None,
        channel_scale: Optional[List[float]] = None,
        preset: Optional[str] = None,
        output_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        if bit_depth not in (8, 10, 12):
            raise ValueError(f"bit_depth must be 8, 10, or 12, got {bit_depth}")

        self.bit_depth = bit_depth
        self.max_value = float((2**bit_depth) - 1)
        self.output_dtype = output_dtype

        # Store flags for control flow (avoids enum comparisons during tracing)
        self.input_is_bgr = input_color == ColorLayout.BGR
        self.output_is_bgr = output_color == ColorLayout.BGR
        self.output_is_grayscale = output_color == ColorLayout.GRAYSCALE
        self.input_is_srgb = input_transfer == TransferFunction.SRGB
        self.input_is_pq = input_transfer == TransferFunction.PQ
        self.input_is_hlg = input_transfer == TransferFunction.HLG
        self.output_is_srgb = output_transfer == TransferFunction.SRGB

        # Use preset if specified
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}"
                )
            preset_config = PRESETS[preset]
            channel_bias = preset_config["bias"]
            channel_scale = preset_config["scale"]

        # Default: no additional normalization
        if channel_bias is None:
            channel_bias = [0.0, 0.0, 0.0]
        if channel_scale is None:
            channel_scale = [1.0, 1.0, 1.0]

        # Register color matrix if gamut conversion needed
        if input_gamut != output_gamut:
            if input_gamut == ColorGamut.BT2020 and output_gamut == ColorGamut.BT709:
                self.register_buffer("color_matrix", BT2020_TO_BT709.to(torch.float16))
            elif input_gamut == ColorGamut.BT709 and output_gamut == ColorGamut.BT2020:
                self.register_buffer("color_matrix", BT709_TO_BT2020.to(torch.float16))
        else:
            self.color_matrix = None

        # Register grayscale weights if needed
        if output_color == ColorLayout.GRAYSCALE:
            self.register_buffer(
                "luminance_weights",
                LUMINANCE_WEIGHTS.view(1, 3, 1, 1).to(torch.float16),
            )
            gray_bias = sum(channel_bias) / 3.0
            gray_scale = sum(channel_scale) / 3.0
            self.register_buffer(
                "bias", torch.tensor([gray_bias]).view(1, 1, 1, 1).to(torch.float16)
            )
            self.register_buffer(
                "scale", torch.tensor([gray_scale]).view(1, 1, 1, 1).to(torch.float16)
            )
        else:
            self.register_buffer(
                "bias", torch.tensor(channel_bias).view(1, 3, 1, 1).to(torch.float16)
            )
            self.register_buffer(
                "scale", torch.tensor(channel_scale).view(1, 3, 1, 1).to(torch.float16)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Handle BGR → RGB if needed
        if self.input_is_bgr:
            x = x.flip(dims=[1])

        # Step 2: Normalize to [0, 1] based on bit depth
        x = x / self.max_value

        # Step 3: Apply inverse transfer function (to linear)
        if self.input_is_srgb:
            x = apply_srgb_inverse(x)
        elif self.input_is_pq:
            x = apply_pq_inverse(x)
        elif self.input_is_hlg:
            x = apply_hlg_inverse(x)

        # Step 4: Apply color gamut conversion if needed
        if self.color_matrix is not None:
            x = apply_gamut_conversion(x, self.color_matrix)

        # Step 5: Apply output transfer function (from linear)
        if self.output_is_srgb:
            x = apply_srgb_gamma(x)

        # Step 6: Convert to grayscale if requested
        if self.output_is_grayscale:
            x = (x * self.luminance_weights).sum(dim=1, keepdim=True)

        # Step 7: Apply normalization (bias and scale)
        x = (x + self.bias) * self.scale

        # Step 8: Convert RGB → BGR if needed
        if self.output_is_bgr:
            x = x.flip(dims=[1])

        # Step 9: Convert to output dtype
        return x.to(self.output_dtype)

    # ==================== Factory Methods ====================
    # These return ExportedProgram ready to lower to any backend

    @classmethod
    def _export(
        cls,
        model: "ImagePreprocessor",
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype,
    ) -> torch.export.ExportedProgram:
        """Helper to export a model to ExportedProgram."""
        # Convert model buffers to match input dtype for consistent precision
        model = model.to(input_dtype)
        model.eval()
        example_inputs = (torch.randn(*shape, dtype=input_dtype),)
        return torch.export.export(model, example_inputs, strict=True)

    @classmethod
    def from_scale_0_1(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float16,
        input_color: ColorLayout = ColorLayout.RGB,
        output_color: ColorLayout = ColorLayout.RGB,
    ) -> torch.export.ExportedProgram:
        """
        Create and export preprocessor that scales [0, 255] → [0, 1].

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype (use float32 for precision)
            output_dtype: Output tensor dtype
            input_color: Input color layout (RGB, BGR)
            output_color: Output color layout (RGB, BGR, GRAYSCALE)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=8,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.LINEAR,
            input_color=input_color,
            output_color=output_color,
            preset="scale_0_1",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_scale_neg1_1(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float16,
        input_color: ColorLayout = ColorLayout.RGB,
        output_color: ColorLayout = ColorLayout.RGB,
    ) -> torch.export.ExportedProgram:
        """
        Create and export preprocessor that scales [0, 255] → [-1, 1].

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype (use float32 for precision)
            output_dtype: Output tensor dtype
            input_color: Input color layout (RGB, BGR)
            output_color: Output color layout (RGB, BGR, GRAYSCALE)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=8,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.LINEAR,
            input_color=input_color,
            output_color=output_color,
            preset="scale_neg1_1",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_imagenet(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float16,
        input_color: ColorLayout = ColorLayout.RGB,
    ) -> torch.export.ExportedProgram:
        """
        Create and export preprocessor with ImageNet normalization.

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            input_color: Input color layout (RGB, BGR)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=8,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.LINEAR,
            input_color=input_color,
            preset="imagenet",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_hdr10(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,  # CoreML uses fp16 internally
        output_dtype: torch.dtype = torch.float16,
        bit_depth: int = 10,
        output_transfer: TransferFunction = TransferFunction.LINEAR,
        output_gamut: ColorGamut = ColorGamut.BT709,
        preset: Optional[str] = None,
    ) -> torch.export.ExportedProgram:
        """
        Create and export preprocessor for HDR10 content.
        HDR10 uses PQ transfer function and BT.2020 color gamut.

        NOTE: CoreML uses fp16 internally. The PQ transfer function has limited
        precision in fp16 due to high-power exponents (m2=78.84). This may cause
        ~5% error in bright regions compared to float32 reference implementations.

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            bit_depth: Input bit depth (10 or 12)
            output_transfer: Output transfer function (LINEAR or SRGB)
            output_gamut: Output color gamut (BT709 or BT2020)
            preset: Optional normalization preset

        Returns:
            ExportedProgram ready to lower to any backend
        """
        if input_dtype == torch.float16:
            warnings.warn(
                "HDR10 (PQ transfer function) has significant precision loss in fp16 "
                "due to high-power exponents (m2=78.84). Expect up to 50% relative error "
                "compared to fp32. Consider using fp32 for input_dtype if accuracy "
                "is critical.",
                UserWarning,
                stacklevel=2,
            )
        model = cls(
            bit_depth=bit_depth,
            input_transfer=TransferFunction.PQ,
            output_transfer=output_transfer,
            input_gamut=ColorGamut.BT2020,
            output_gamut=output_gamut,
            preset=preset,
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_hlg(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,  # CoreML uses fp16 internally
        output_dtype: torch.dtype = torch.float16,
        bit_depth: int = 10,
        output_transfer: TransferFunction = TransferFunction.LINEAR,
        output_gamut: ColorGamut = ColorGamut.BT709,
        preset: Optional[str] = None,
    ) -> torch.export.ExportedProgram:
        """
        Create and export preprocessor for HLG (Hybrid Log-Gamma) content.
        HLG is used in broadcast HDR and typically uses BT.2020 color gamut.

        NOTE: CoreML uses fp16 internally, which may affect precision.

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            bit_depth: Input bit depth (10 or 12)
            output_transfer: Output transfer function (LINEAR or SRGB)
            output_gamut: Output color gamut (BT709 or BT2020)
            preset: Optional normalization preset

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=bit_depth,
            input_transfer=TransferFunction.HLG,
            output_transfer=output_transfer,
            input_gamut=ColorGamut.BT2020,
            output_gamut=output_gamut,
            preset=preset,
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_sdr(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float16,
        normalize_to_linear: bool = False,
        preset: Optional[str] = "scale_0_1",
    ) -> torch.export.ExportedProgram:
        """
        Create and export preprocessor for standard SDR content.
        SDR uses sRGB gamma and BT.709 color gamut.

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            normalize_to_linear: If True, convert sRGB gamma to linear
            preset: Normalization preset (scale_0_1, scale_neg1_1, imagenet)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=8,
            input_transfer=(
                TransferFunction.SRGB
                if normalize_to_linear
                else TransferFunction.LINEAR
            ),
            output_transfer=TransferFunction.LINEAR,
            input_gamut=ColorGamut.BT709,
            output_gamut=ColorGamut.BT709,
            preset=preset,
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)


# =============================================================================
# ImagePostprocessor
# =============================================================================


class ImagePostprocessor(torch.nn.Module):
    """
    Convert model output tensor back to displayable image format.
    Inverse of ImagePreprocessor.

    Processing pipeline (inverse of preprocessor):
    1. Convert input color layout (BGR → RGB if needed)
    2. Apply inverse normalization: x = (x / scale) - bias
    3. Convert from grayscale to RGB if needed
    4. Apply inverse output transfer function (sRGB → linear if needed)
    5. Apply inverse color gamut conversion (BT.709 → BT.2020 if needed)
    6. Apply forward transfer function (linear → gamma/PQ/HLG if needed)
    7. Scale to [0, max_value] based on bit depth
    8. Convert output color layout (RGB → BGR if needed)

    Args:
        bit_depth: Output bit depth (8, 10, or 12). Default 8 for SDR.
        input_transfer: Transfer function of model output (SRGB, LINEAR).
        output_transfer: Desired transfer function of output (SRGB, PQ, HLG, LINEAR).
        input_gamut: Color gamut of model output (BT709, BT2020). Default BT709.
        output_gamut: Desired color gamut of output (BT709, BT2020). Default BT709.
        input_color: Color layout of model output (RGB, BGR). Default RGB.
        output_color: Desired color layout (RGB, BGR). Default RGB.
        channel_bias: Per-channel bias used by preprocessor (will be inverted).
        channel_scale: Per-channel scale used by preprocessor (will be inverted).
        preset: Optional preset name matching preprocessor preset.
        output_dtype: Output data type (torch.float16 or torch.float32).

    Input:
        float16/float32 tensor [B, C, H, W] - model output (normalized)

    Output:
        float16 tensor [B, C, H, W] with values in [0, max_val] based on bit_depth
        Ready for uint8 conversion via vDSP.
    """

    def __init__(
        self,
        bit_depth: int = 8,
        input_transfer: TransferFunction = TransferFunction.LINEAR,
        output_transfer: TransferFunction = TransferFunction.LINEAR,
        input_gamut: ColorGamut = ColorGamut.BT709,
        output_gamut: ColorGamut = ColorGamut.BT709,
        input_color: ColorLayout = ColorLayout.RGB,
        output_color: ColorLayout = ColorLayout.RGB,
        channel_bias: Optional[List[float]] = None,
        channel_scale: Optional[List[float]] = None,
        preset: Optional[str] = None,
        output_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        if bit_depth not in (8, 10, 12):
            raise ValueError(f"bit_depth must be 8, 10, or 12, got {bit_depth}")

        self.bit_depth = bit_depth
        self.max_value = float((2**bit_depth) - 1)
        self.output_dtype = output_dtype

        # Store flags for control flow
        self.input_is_bgr = input_color == ColorLayout.BGR
        self.output_is_bgr = output_color == ColorLayout.BGR
        self.input_is_srgb = input_transfer == TransferFunction.SRGB
        self.output_is_srgb = output_transfer == TransferFunction.SRGB
        self.output_is_pq = output_transfer == TransferFunction.PQ
        self.output_is_hlg = output_transfer == TransferFunction.HLG

        # Use preset if specified
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}"
                )
            preset_config = PRESETS[preset]
            channel_bias = preset_config["bias"]
            channel_scale = preset_config["scale"]

        # Default: no normalization to invert
        if channel_bias is None:
            channel_bias = [0.0, 0.0, 0.0]
        if channel_scale is None:
            channel_scale = [1.0, 1.0, 1.0]

        # Register color matrix if gamut conversion needed (inverse direction)
        if input_gamut != output_gamut:
            if input_gamut == ColorGamut.BT709 and output_gamut == ColorGamut.BT2020:
                self.register_buffer("color_matrix", BT709_TO_BT2020.to(torch.float16))
            elif input_gamut == ColorGamut.BT2020 and output_gamut == ColorGamut.BT709:
                self.register_buffer("color_matrix", BT2020_TO_BT709.to(torch.float16))
        else:
            self.color_matrix = None

        # Register bias and scale for inverse normalization
        self.register_buffer(
            "bias", torch.tensor(channel_bias).view(1, 3, 1, 1).to(torch.float16)
        )
        self.register_buffer(
            "scale", torch.tensor(channel_scale).view(1, 3, 1, 1).to(torch.float16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Handle BGR → RGB if needed
        if self.input_is_bgr:
            x = x.flip(dims=[1])

        # Step 2: Inverse normalization: x = (x / scale) - bias
        x = (x / self.scale) - self.bias

        # Step 3: Apply inverse input transfer function (if model output was in sRGB)
        if self.input_is_srgb:
            x = apply_srgb_inverse(x)

        # Step 4: Apply color gamut conversion if needed
        if self.color_matrix is not None:
            x = apply_gamut_conversion(x, self.color_matrix)

        # Step 5: Apply output transfer function (to target gamma)
        if self.output_is_srgb:
            x = apply_srgb_gamma(x)
        elif self.output_is_pq:
            x = apply_pq_forward(x)
        elif self.output_is_hlg:
            x = apply_hlg_forward(x)

        # Step 6: Scale to [0, max_value] based on bit depth
        x = x * self.max_value

        # Step 7: Clamp to valid range
        x = x.clamp(min=0.0, max=self.max_value)

        # Step 8: Convert RGB → BGR if needed
        if self.output_is_bgr:
            x = x.flip(dims=[1])

        # Step 9: Convert to output dtype
        return x.to(self.output_dtype)

    # ==================== Factory Methods ====================
    # These return ExportedProgram ready to lower to any backend

    @classmethod
    def _export(
        cls,
        model: "ImagePostprocessor",
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype,
    ) -> torch.export.ExportedProgram:
        """Helper to export a model to ExportedProgram."""
        # Convert model buffers to match input dtype for consistent precision
        model = model.to(input_dtype)
        model.eval()
        example_inputs = (torch.randn(*shape, dtype=input_dtype),)
        return torch.export.export(model, example_inputs, strict=True)

    @classmethod
    def from_scale_0_1(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float16,
        input_color: ColorLayout = ColorLayout.RGB,
        output_color: ColorLayout = ColorLayout.RGB,
    ) -> torch.export.ExportedProgram:
        """
        Create and export postprocessor for model output in [0, 1] range.
        Converts [0, 1] → [0, 255].

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            input_color: Input color layout (RGB, BGR)
            output_color: Output color layout (RGB, BGR)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=8,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.LINEAR,
            input_color=input_color,
            output_color=output_color,
            preset="scale_0_1",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_scale_neg1_1(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float16,
        input_color: ColorLayout = ColorLayout.RGB,
        output_color: ColorLayout = ColorLayout.RGB,
    ) -> torch.export.ExportedProgram:
        """
        Create and export postprocessor for model output in [-1, 1] range.
        Converts [-1, 1] → [0, 255].
        Common for GANs and diffusion models.

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            input_color: Input color layout (RGB, BGR)
            output_color: Output color layout (RGB, BGR)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=8,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.LINEAR,
            input_color=input_color,
            output_color=output_color,
            preset="scale_neg1_1",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_imagenet(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float16,
        output_color: ColorLayout = ColorLayout.RGB,
    ) -> torch.export.ExportedProgram:
        """
        Create and export postprocessor for ImageNet-normalized model output.
        Inverts ImageNet normalization → [0, 255].

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            output_color: Output color layout (RGB, BGR)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=8,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.LINEAR,
            output_color=output_color,
            preset="imagenet",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_linear_to_srgb(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,
        output_dtype: torch.dtype = torch.float16,
        input_color: ColorLayout = ColorLayout.RGB,
        output_color: ColorLayout = ColorLayout.RGB,
    ) -> torch.export.ExportedProgram:
        """
        Create and export postprocessor that converts linear [0, 1] to sRGB [0, 255].
        Useful for HDR models that output linear light.

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            input_color: Input color layout (RGB, BGR)
            output_color: Output color layout (RGB, BGR)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=8,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.SRGB,
            input_color=input_color,
            output_color=output_color,
            preset="scale_0_1",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_linear_to_hdr10(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float16,  # CoreML uses fp16 internally
        output_dtype: torch.dtype = torch.float16,
        bit_depth: int = 10,
        input_gamut: ColorGamut = ColorGamut.BT709,
    ) -> torch.export.ExportedProgram:
        """
        Create and export postprocessor that converts linear [0, 1] to HDR10.
        Outputs PQ-encoded BT.2020 content.

        NOTE: CoreML uses fp16 internally, which may affect PQ precision.

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype
            output_dtype: Output tensor dtype
            bit_depth: Output bit depth (10 or 12)
            input_gamut: Input color gamut (BT709 or BT2020)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        if input_dtype == torch.float16:
            warnings.warn(
                "HDR10 (PQ transfer function) has significant precision loss in fp16 "
                "due to high-power exponents (m2=78.84). Expect up to 50% relative error "
                "compared to fp32. Consider using fp32 for input_dtype if accuracy "
                "is critical.",
                UserWarning,
                stacklevel=2,
            )
        model = cls(
            bit_depth=bit_depth,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.PQ,
            input_gamut=input_gamut,
            output_gamut=ColorGamut.BT2020,
            preset="scale_0_1",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)

    @classmethod
    def from_linear_to_hlg(
        cls,
        shape: Tuple[int, int, int, int],
        input_dtype: torch.dtype = torch.float32,  # float32 recommended for HDR
        output_dtype: torch.dtype = torch.float16,
        bit_depth: int = 10,
        input_gamut: ColorGamut = ColorGamut.BT709,
    ) -> torch.export.ExportedProgram:
        """
        Create and export postprocessor that converts linear [0, 1] to HLG.
        Outputs HLG-encoded BT.2020 content for broadcast HDR.

        NOTE: float32 input_dtype is recommended for accurate HLG calculations.

        Args:
            shape: Input shape (batch, channels, height, width)
            input_dtype: Input tensor dtype (float32 recommended for HDR)
            output_dtype: Output tensor dtype
            bit_depth: Output bit depth (10 or 12)
            input_gamut: Input color gamut (BT709 or BT2020)

        Returns:
            ExportedProgram ready to lower to any backend
        """
        model = cls(
            bit_depth=bit_depth,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.HLG,
            input_gamut=input_gamut,
            output_gamut=ColorGamut.BT2020,
            preset="scale_0_1",
            output_dtype=output_dtype,
        )
        return cls._export(model, shape, input_dtype)
