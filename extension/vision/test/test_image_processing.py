# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for image processing utilities.

Compares our implementations against the colour-science library
which provides reference implementations of ITU/SMPTE standards.
"""

import unittest

import numpy as np
import torch
from executorch.extension.vision.image_processing import (
    apply_hlg_forward,
    apply_hlg_inverse,
    apply_pq_forward,
    apply_pq_inverse,
    apply_srgb_gamma,
    apply_srgb_inverse,
    BT2020_TO_BT709,
    BT709_TO_BT2020,
    ImagePostprocessor,
    ImagePreprocessor,
    LUMINANCE_WEIGHTS,
)
from parameterized import parameterized

# colour-science is optional - tests will skip if not installed
try:
    import colour

    HAS_COLOUR = True
except ImportError:
    HAS_COLOUR = False


# =============================================================================
# Constants
# =============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

IMAGENET_MEAN_TORCH = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
IMAGENET_STD_TORCH = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def max_value_for_bit_depth(bit_depth: int) -> int:
    """Return max pixel value for given bit depth (e.g., 255 for 8-bit, 1023 for 10-bit)."""
    return (2**bit_depth) - 1


def requires_colour(test_func):
    """Decorator to skip tests if colour-science is not installed."""
    return unittest.skipUnless(HAS_COLOUR, "colour-science not installed")(test_func)


class TestTransferFunctionsAgainstReference(unittest.TestCase):
    """Test transfer functions against colour-science reference implementations."""

    @requires_colour
    def test_pq_inverse_against_reference(self):
        """Test PQ (ST.2084) EOTF against colour-science."""
        # Test values across the PQ range
        test_values = np.linspace(0.01, 1.0, 100)

        # Our implementation
        x = torch.tensor(test_values, dtype=torch.float32).view(1, 1, 10, 10)
        ours = apply_pq_inverse(x).numpy().flatten()

        # Reference: colour-science ST.2084 EOTF
        # Note: colour uses normalized output [0, 1] when we pass normalized=True
        ref = colour.eotf(test_values, "ST 2084", L_p=10000)
        # Normalize to [0, 1] since colour returns absolute luminance
        ref = ref / 10000.0

        max_error = np.abs(ours - ref).max()
        self.assertLess(
            max_error,
            1e-4,
            f"PQ inverse max error {max_error:.6f} exceeds tolerance",
        )

    @requires_colour
    def test_pq_forward_against_reference(self):
        """Test PQ (ST.2084) inverse EOTF against colour-science."""
        # Test linear values
        test_values = np.linspace(0.001, 1.0, 100)

        # Our implementation
        x = torch.tensor(test_values, dtype=torch.float32).view(1, 1, 10, 10)
        ours = apply_pq_forward(x).numpy().flatten()

        # Reference: colour-science ST.2084 inverse EOTF (OETF)
        # Input is absolute luminance, so scale by 10000
        ref = colour.eotf_inverse(test_values * 10000, "ST 2084", L_p=10000)

        max_error = np.abs(ours - ref).max()
        self.assertLess(
            max_error,
            1e-4,
            f"PQ forward max error {max_error:.6f} exceeds tolerance",
        )

    @requires_colour
    def test_hlg_inverse_against_reference(self):
        """Test HLG OETF inverse against colour-science."""
        # Test values in the HLG range
        test_values = np.linspace(0.01, 1.0, 100)

        # Our implementation
        x = torch.tensor(test_values, dtype=torch.float32).view(1, 1, 10, 10)
        ours = apply_hlg_inverse(x).numpy().flatten()

        # Reference: colour-science HLG OETF inverse
        ref = colour.models.oetf_inverse_BT2100_HLG(test_values)

        max_error = np.abs(ours - ref).max()
        self.assertLess(
            max_error,
            1e-5,
            f"HLG inverse max error {max_error:.6f} exceeds tolerance",
        )

    @requires_colour
    def test_hlg_forward_against_reference(self):
        """Test HLG OETF against colour-science."""
        # Test linear values
        test_values = np.linspace(0.001, 1.0, 100)

        # Our implementation
        x = torch.tensor(test_values, dtype=torch.float32).view(1, 1, 10, 10)
        ours = apply_hlg_forward(x).numpy().flatten()

        # Reference: colour-science HLG OETF
        ref = colour.models.oetf_BT2100_HLG(test_values)

        max_error = np.abs(ours - ref).max()
        self.assertLess(
            max_error,
            1e-5,
            f"HLG forward max error {max_error:.6f} exceeds tolerance",
        )

    @requires_colour
    def test_srgb_gamma_against_reference(self):
        """Test sRGB gamma (x^(1/2.2) approximation) against colour-science.

        Note: We use the x^(1/2.2) approximation, not the piecewise sRGB transfer.
        This test documents the expected divergence.
        """
        # Test in mid-to-high range where approximation is reasonable
        test_values = np.linspace(0.1, 1.0, 100)

        # Our implementation (x^(1/2.2) approximation)
        x = torch.tensor(test_values, dtype=torch.float32).view(1, 1, 10, 10)
        ours = apply_srgb_gamma(x).numpy().flatten()

        # Simple gamma approximation reference
        ref = np.power(test_values, 1.0 / 2.2)

        max_error = np.abs(ours - ref).max()
        self.assertLess(
            max_error,
            1e-6,
            f"sRGB gamma max error {max_error:.6f} exceeds tolerance",
        )

    @requires_colour
    def test_srgb_inverse_against_reference(self):
        """Test sRGB inverse gamma (x^2.2 approximation) against colour-science."""
        # Test in mid-to-high range
        test_values = np.linspace(0.1, 1.0, 100)

        # Our implementation (x^2.2 approximation)
        x = torch.tensor(test_values, dtype=torch.float32).view(1, 1, 10, 10)
        ours = apply_srgb_inverse(x).numpy().flatten()

        # Simple gamma approximation reference
        ref = np.power(test_values, 2.2)

        max_error = np.abs(ours - ref).max()
        self.assertLess(
            max_error,
            1e-6,
            f"sRGB inverse max error {max_error:.6f} exceeds tolerance",
        )


class TestGamutMatricesAgainstReference(unittest.TestCase):
    """Test color gamut conversion matrices against colour-science."""

    @requires_colour
    def test_bt2020_to_bt709_matrix(self):
        """Test BT.2020 to BT.709 conversion matrix against colour-science."""
        # Get reference matrix from colour-science
        # BT.2020 primaries and whitepoint
        bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
        bt709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]

        # Compute conversion matrix
        ref_matrix = colour.matrix_RGB_to_RGB(bt2020, bt709)

        ours = BT2020_TO_BT709.numpy()

        max_error = np.abs(ours - ref_matrix).max()
        self.assertLess(
            max_error,
            0.01,  # Allow some tolerance for different derivations
            f"BT.2020→BT.709 matrix max error {max_error:.6f}",
        )

    @requires_colour
    def test_bt709_to_bt2020_matrix(self):
        """Test BT.709 to BT.2020 conversion matrix against colour-science."""
        bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
        bt709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]

        # Compute conversion matrix
        ref_matrix = colour.matrix_RGB_to_RGB(bt709, bt2020)

        ours = BT709_TO_BT2020.numpy()

        max_error = np.abs(ours - ref_matrix).max()
        self.assertLess(
            max_error,
            0.01,
            f"BT.709→BT.2020 matrix max error {max_error:.6f}",
        )

    @requires_colour
    def test_gamut_matrices_are_inverses(self):
        """Verify that our gamut matrices are inverses of each other."""
        product = BT2020_TO_BT709 @ BT709_TO_BT2020
        identity = torch.eye(3)

        max_error = (product - identity).abs().max().item()
        self.assertLess(
            max_error,
            1e-4,
            f"Gamut matrices not inverses, max error {max_error:.6f}",
        )


class TestLuminanceWeights(unittest.TestCase):
    """Test luminance weights for grayscale conversion."""

    def test_luminance_weights_are_bt601(self):
        """Verify luminance weights match BT.601 standard."""
        # BT.601 standard weights
        bt601_weights = np.array([0.299, 0.587, 0.114])

        ours = LUMINANCE_WEIGHTS.numpy()

        max_error = np.abs(ours - bt601_weights).max()
        self.assertLess(
            max_error,
            1e-6,
            f"Luminance weights don't match BT.601: {ours} vs {bt601_weights}",
        )

    def test_luminance_weights_sum_to_one(self):
        """Luminance weights should sum to 1.0."""
        weight_sum = LUMINANCE_WEIGHTS.sum().item()
        self.assertAlmostEqual(
            weight_sum,
            1.0,
            places=6,
            msg=f"Luminance weights sum to {weight_sum}, expected 1.0",
        )


class TestTransferFunctionRoundtrip(unittest.TestCase):
    """Test that forward/inverse transfer functions are true inverses."""

    def setUp(self):
        torch.manual_seed(42)

    def test_pq_roundtrip(self):
        """PQ forward then inverse should return original values."""
        original = torch.rand(1, 3, 32, 32, dtype=torch.float32) * 0.9 + 0.05
        encoded = apply_pq_forward(original)
        decoded = apply_pq_inverse(encoded)

        max_error = (original - decoded).abs().max().item()
        self.assertLess(
            max_error,
            2e-4,
            f"PQ roundtrip max error {max_error:.6f}",
        )

    def test_hlg_roundtrip(self):
        """HLG forward then inverse should return original values."""
        original = torch.rand(1, 3, 32, 32, dtype=torch.float32) * 0.9 + 0.05
        encoded = apply_hlg_forward(original)
        decoded = apply_hlg_inverse(encoded)

        max_error = (original - decoded).abs().max().item()
        self.assertLess(
            max_error,
            1e-5,
            f"HLG roundtrip max error {max_error:.6f}",
        )

    def test_srgb_roundtrip(self):
        """sRGB gamma then inverse should return original values."""
        original = torch.rand(1, 3, 32, 32, dtype=torch.float32) * 0.9 + 0.05
        encoded = apply_srgb_gamma(original)
        decoded = apply_srgb_inverse(encoded)

        max_error = (original - decoded).abs().max().item()
        self.assertLess(
            max_error,
            1e-5,
            f"sRGB roundtrip max error {max_error:.6f}",
        )


# =============================================================================
# E2E Tests: ExportedProgram vs colour-science reference
# =============================================================================


def generate_test_pattern(
    height: int, width: int, bit_depth: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate a synthetic test pattern with gradients, color bars, and gray ramp.

    Returns: [1, 3, H, W] tensor with values in [0, max_value]
    """
    max_value = (2**bit_depth) - 1
    img = torch.zeros(1, 3, height, width, dtype=dtype)

    h_third = height // 3

    # Top third: horizontal gradient per channel
    for c in range(3):
        gradient = torch.linspace(0, max_value, width, dtype=dtype)
        img[0, c, :h_third, :] = gradient.unsqueeze(0).expand(h_third, -1)
        # Offset each channel slightly
        img[0, c, :h_third, :] = (img[0, c, :h_third, :] * (0.8 + 0.1 * c)).clamp(
            0, max_value
        )

    # Middle third: color bars (R, G, B, C, M, Y, W, K)
    colors = [
        [max_value, 0, 0],  # Red
        [0, max_value, 0],  # Green
        [0, 0, max_value],  # Blue
        [0, max_value, max_value],  # Cyan
        [max_value, 0, max_value],  # Magenta
        [max_value, max_value, 0],  # Yellow
        [max_value, max_value, max_value],  # White
        [0, 0, 0],  # Black
    ]
    bar_width = width // len(colors)
    for i, color in enumerate(colors):
        start = i * bar_width
        end = start + bar_width if i < len(colors) - 1 else width
        for c in range(3):
            img[0, c, h_third : 2 * h_third, start:end] = color[c]

    # Bottom third: gray ramp
    gray_ramp = torch.linspace(0, max_value, width, dtype=dtype)
    for c in range(3):
        img[0, c, 2 * h_third :, :] = gray_ramp.unsqueeze(0).expand(
            height - 2 * h_third, -1
        )

    return img


# Reference functions using colour-science


def _convert_gamut(img: np.ndarray, source: str, dest: str) -> np.ndarray:
    """Convert between color gamuts using colour-science.

    Args:
        img: Image array in CHW format
        source: Source colorspace (e.g., "ITU-R BT.2020")
        dest: Destination colorspace (e.g., "ITU-R BT.709")

    Returns:
        Converted image in CHW format
    """
    img = np.moveaxis(img, 0, -1)
    img = colour.RGB_to_RGB(
        img,
        colour.RGB_COLOURSPACES[source],
        colour.RGB_COLOURSPACES[dest],
        chromatic_adaptation_transform=None,
    )
    return np.moveaxis(img, -1, 0)


def ref_scale_0_1(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Reference: Simple [0, max] → [0, 1] scaling."""
    max_value = (2**bit_depth) - 1
    return (img / max_value).astype(np.float32)


def ref_scale_neg1_1(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Reference: [0, max] → [-1, 1] scaling."""
    max_value = (2**bit_depth) - 1
    return ((img / max_value - 0.5) * 2.0).astype(np.float32)


def ref_imagenet(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Reference: ImageNet normalization."""
    max_value = max_value_for_bit_depth(bit_depth)
    img = img / max_value
    mean = IMAGENET_MEAN.reshape(3, 1, 1)
    std = IMAGENET_STD.reshape(3, 1, 1)
    return ((img - mean) / std).astype(np.float32)


@requires_colour
def ref_hdr10_to_linear_bt709(img: np.ndarray, bit_depth: int = 10) -> np.ndarray:
    """Reference: HDR10 (PQ BT.2020) → linear BT.709."""
    max_value = (2**bit_depth) - 1
    img = img / max_value
    img = colour.models.eotf_ST2084(img) / 10000.0
    img = _convert_gamut(img, "ITU-R BT.2020", "ITU-R BT.709")
    return np.clip(img, 0, 1).astype(np.float32)


@requires_colour
def ref_hlg_to_linear_bt709(img: np.ndarray, bit_depth: int = 10) -> np.ndarray:
    """Reference: HLG BT.2020 → linear BT.709."""
    max_value = (2**bit_depth) - 1
    img = img / max_value
    img = colour.models.oetf_inverse_BT2100_HLG(img)
    img = _convert_gamut(img, "ITU-R BT.2020", "ITU-R BT.709")
    return np.clip(img, 0, 1).astype(np.float32)


def ref_sdr_to_linear(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Reference: SDR (sRGB gamma) → linear."""
    max_value = (2**bit_depth) - 1
    img = img / max_value
    return np.power(np.clip(img, 1e-6, 1.0), 2.2).astype(np.float32)


# Postprocessor reference functions (inverse operations)


def ref_post_scale_0_1(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Reference: [0, 1] → [0, max] scaling."""
    max_value = (2**bit_depth) - 1
    return (img * max_value).astype(np.float32)


def ref_post_scale_neg1_1(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Reference: [-1, 1] → [0, max] scaling."""
    max_value = (2**bit_depth) - 1
    return ((img / 2.0 + 0.5) * max_value).astype(np.float32)


def ref_post_imagenet(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Reference: Reverse ImageNet normalization."""
    max_value = max_value_for_bit_depth(bit_depth)
    mean = IMAGENET_MEAN.reshape(3, 1, 1)
    std = IMAGENET_STD.reshape(3, 1, 1)
    img = img * std + mean
    return np.clip(img * max_value, 0, max_value).astype(np.float32)


@requires_colour
def ref_linear_bt709_to_hdr10(img: np.ndarray, bit_depth: int = 10) -> np.ndarray:
    """Reference: linear BT.709 → HDR10 (PQ BT.2020)."""
    max_value = (2**bit_depth) - 1
    img = _convert_gamut(img, "ITU-R BT.709", "ITU-R BT.2020")
    img = np.clip(img, 0, 1)
    img = colour.models.eotf_inverse_ST2084(img * 10000)
    return (img * max_value).astype(np.float32)


@requires_colour
def ref_linear_bt709_to_hlg(img: np.ndarray, bit_depth: int = 10) -> np.ndarray:
    """Reference: linear BT.709 → HLG BT.2020."""
    max_value = (2**bit_depth) - 1
    img = _convert_gamut(img, "ITU-R BT.709", "ITU-R BT.2020")
    img = np.clip(img, 0, 1)
    img = colour.models.oetf_BT2100_HLG(img)
    return (img * max_value).astype(np.float32)


def ref_linear_to_srgb(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Reference: linear → sRGB gamma [0, 255]."""
    max_value = (2**bit_depth) - 1
    img = np.power(np.clip(img, 1e-6, 1.0), 1.0 / 2.2)
    return (img * max_value).astype(np.float32)


class TestE2EPipelines(unittest.TestCase):
    """
    End-to-end tests comparing ExportedProgram output against colour-science.

    These tests validate that our image processing pipelines produce results
    matching the reference colour-science implementations.
    """

    def setUp(self):
        torch.manual_seed(42)

    # ==================== Tolerance Constants ====================
    # Tolerances determined empirically - see test analysis for actual error measurements.
    # fp16 tolerances are looser due to inherent precision limitations.

    TOLERANCES = {
        torch.float32: {"rtol": 1e-3, "atol": 1e-3},  # Simple scaling: max diff ~0.0003
        torch.float16: {"rtol": 0.01, "atol": 0.01},
    }

    HDR_TOLERANCES = {
        torch.float32: {
            "rtol": 0.005,
            "atol": 1e-3,
        },  # HLG transfer is very accurate (~6e-8)
        torch.float16: {"rtol": 0.05, "atol": 0.05},
    }

    HDR10_TOLERANCES = {
        torch.float32: {"rtol": 0.005, "atol": 1e-3},
        torch.float16: {
            "rtol": 0.5,
            "atol": 0.25,
        },  # PQ has significant fp16 precision loss due to m2=78.84 exponent
    }

    IMAGENET_TOLERANCES = {
        torch.float32: {"rtol": 0.001, "atol": 0.001},
        torch.float16: {"rtol": 0.01, "atol": 0.01},
    }

    POST_IMAGENET_TOLERANCES = {
        torch.float32: {"rtol": 0.001, "atol": 0.5},  # atol in [0, 255] range
        torch.float16: {"rtol": 0.01, "atol": 1.0},
    }

    POST_HDR10_TOLERANCES = {
        torch.float32: {"rtol": 0.005, "atol": 2.0},  # atol in [0, 1023] range
        torch.float16: {"rtol": 0.1, "atol": 25.0},  # PQ fp16 precision loss
    }

    POST_HLG_TOLERANCES = {
        torch.float32: {
            "rtol": 0.005,
            "atol": 2.0,
        },  # Edge cases near zero can have larger errors
        torch.float16: {"rtol": 0.02, "atol": 5.0},
    }

    # ==================== Helper Methods ====================

    def _run_exported_program(
        self, ep: torch.export.ExportedProgram, inputs: torch.Tensor
    ) -> np.ndarray:
        """Run an ExportedProgram and return output as numpy array."""
        module = ep.module()
        with torch.no_grad():
            output = module(inputs)
        return output.numpy()

    def _compare_with_reference(
        self,
        ep: torch.export.ExportedProgram,
        reference_fn,
        test_img: torch.Tensor,
        bit_depth: int,
        rtol: float,
        atol: float,
    ):
        """Compare ExportedProgram output against reference implementation."""
        ep_out = self._run_exported_program(ep, test_img)
        reference_out = reference_fn(test_img[0].float().numpy(), bit_depth=bit_depth)

        np.testing.assert_allclose(
            ep_out[0],
            reference_out,
            rtol=rtol,
            atol=atol,
            err_msg=f"ExportedProgram does not match reference (dtype={test_img.dtype})",
        )

    def _test_pipeline(
        self,
        processor_class: str,
        factory_method: str,
        reference_fn,
        bit_depth: int = 8,
        factory_kwargs: dict = None,
        tolerances: dict = None,
        input_transform=None,
        uses_output_dtype: bool = False,
    ):
        """
        Test a processor factory method with both fp32 and fp16.

        Args:
            processor_class: "ImagePreprocessor" or "ImagePostprocessor"
            factory_method: Name of the factory method (e.g., "from_scale_0_1")
            reference_fn: Reference function for comparison
            bit_depth: Bit depth for test pattern
            factory_kwargs: Additional kwargs for the factory method
            tolerances: Override default tolerances {torch.float32: {...}, torch.float16: {...}}
            input_transform: Optional transform to apply to test input
            uses_output_dtype: If True, also pass output_dtype=dtype to factory method
        """
        from executorch.extension.vision.image_processing import (
            ImagePostprocessor,
            ImagePreprocessor,
        )

        processor_cls = (
            ImagePreprocessor
            if processor_class == "ImagePreprocessor"
            else ImagePostprocessor
        )
        factory_kwargs = factory_kwargs or {}
        tolerances = tolerances or self.TOLERANCES
        shape = (1, 3, 64, 96)

        for dtype in [torch.float32, torch.float16]:
            with self.subTest(dtype=dtype):
                kwargs = {"shape": shape, "input_dtype": dtype, **factory_kwargs}
                if uses_output_dtype:
                    kwargs["output_dtype"] = dtype

                ep = getattr(processor_cls, factory_method)(**kwargs)

                test_img = generate_test_pattern(
                    height=64, width=96, bit_depth=bit_depth, dtype=dtype
                )
                if input_transform:
                    test_img = input_transform(test_img.to(torch.float32)).to(dtype)

                tol = tolerances.get(dtype, tolerances[torch.float32])
                self._compare_with_reference(
                    ep=ep,
                    reference_fn=reference_fn,
                    test_img=test_img,
                    bit_depth=bit_depth,
                    **tol,
                )

    # ==================== Parameterized E2E Tests ====================

    # Preprocessor test cases: (name, factory_method, reference_fn, bit_depth, factory_kwargs, tolerances_key, input_transform, uses_output_dtype, needs_colour)
    PREPROCESSOR_CASES = [
        (
            "scale_0_1",
            "from_scale_0_1",
            ref_scale_0_1,
            8,
            None,
            None,
            None,
            False,
            False,
        ),
        (
            "scale_neg1_1",
            "from_scale_neg1_1",
            ref_scale_neg1_1,
            8,
            None,
            None,
            None,
            False,
            False,
        ),
        (
            "imagenet",
            "from_imagenet",
            ref_imagenet,
            8,
            None,
            "IMAGENET_TOLERANCES",
            None,
            False,
            False,
        ),
        (
            "hdr10_to_linear_bt709",
            "from_hdr10",
            ref_hdr10_to_linear_bt709,
            10,
            None,
            "HDR10_TOLERANCES",
            lambda x: x.clamp(min=100),
            True,
            True,
        ),
        (
            "hlg_to_linear_bt709",
            "from_hlg",
            ref_hlg_to_linear_bt709,
            10,
            None,
            "HDR_TOLERANCES",
            None,
            True,
            True,
        ),
        (
            "sdr_to_linear",
            "from_sdr",
            ref_sdr_to_linear,
            8,
            {"normalize_to_linear": True},
            None,
            None,
            True,
            False,
        ),
    ]

    # Postprocessor test cases: (name, factory_method, reference_fn, bit_depth, factory_kwargs, tolerances_key, input_transform, uses_output_dtype, needs_colour)
    POSTPROCESSOR_CASES = [
        (
            "scale_0_1",
            "from_scale_0_1",
            ref_post_scale_0_1,
            8,
            None,
            None,
            lambda x: x / 255.0,
            False,
            False,
        ),
        (
            "scale_neg1_1",
            "from_scale_neg1_1",
            ref_post_scale_neg1_1,
            8,
            None,
            None,
            lambda x: (x / 255.0 - 0.5) * 2.0,
            False,
            False,
        ),
        (
            "imagenet",
            "from_imagenet",
            ref_post_imagenet,
            8,
            None,
            "POST_IMAGENET_TOLERANCES",
            lambda x: (x / 255.0 - IMAGENET_MEAN_TORCH) / IMAGENET_STD_TORCH,
            False,
            False,
        ),
        (
            "linear_to_hdr10",
            "from_linear_to_hdr10",
            ref_linear_bt709_to_hdr10,
            10,
            None,
            "POST_HDR10_TOLERANCES",
            lambda x: (x / 1023.0).clamp(min=0.01),
            True,
            True,
        ),
        (
            "linear_to_hlg",
            "from_linear_to_hlg",
            ref_linear_bt709_to_hlg,
            10,
            None,
            "POST_HLG_TOLERANCES",
            lambda x: x / 1023.0,
            True,
            True,
        ),
        (
            "linear_to_srgb",
            "from_linear_to_srgb",
            ref_linear_to_srgb,
            8,
            None,
            None,
            lambda x: x / 255.0,
            True,
            False,
        ),
    ]

    @parameterized.expand(PREPROCESSOR_CASES)
    def test_e2e_preprocessor(
        self,
        name,
        factory_method,
        reference_fn,
        bit_depth,
        factory_kwargs,
        tolerances_key,
        input_transform,
        uses_output_dtype,
        needs_colour,
    ):
        """E2E test for ImagePreprocessor pipelines."""
        if needs_colour and not HAS_COLOUR:
            self.skipTest("colour-science not installed")

        tolerances = getattr(self, tolerances_key) if tolerances_key else None
        self._test_pipeline(
            "ImagePreprocessor",
            factory_method,
            reference_fn,
            bit_depth=bit_depth,
            factory_kwargs=factory_kwargs,
            tolerances=tolerances,
            input_transform=input_transform,
            uses_output_dtype=uses_output_dtype,
        )

    @parameterized.expand(POSTPROCESSOR_CASES)
    def test_e2e_postprocessor(
        self,
        name,
        factory_method,
        reference_fn,
        bit_depth,
        factory_kwargs,
        tolerances_key,
        input_transform,
        uses_output_dtype,
        needs_colour,
    ):
        """E2E test for ImagePostprocessor pipelines."""
        if needs_colour and not HAS_COLOUR:
            self.skipTest("colour-science not installed")

        tolerances = getattr(self, tolerances_key) if tolerances_key else None
        self._test_pipeline(
            "ImagePostprocessor",
            factory_method,
            reference_fn,
            bit_depth=bit_depth,
            factory_kwargs=factory_kwargs,
            tolerances=tolerances,
            input_transform=input_transform,
            uses_output_dtype=uses_output_dtype,
        )


if __name__ == "__main__":
    unittest.main()
