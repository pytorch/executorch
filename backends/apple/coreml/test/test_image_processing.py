# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

"""
Tests for CoreML image preprocessing models.

These tests serve as reference examples for how to deploy ImagePreprocessor
and ImagePostprocessor to CoreML. Each test demonstrates the recommended
dtype and compute precision settings for different use cases.

## Dtype and Compute Precision Guidelines

### Pattern 1: Full fp16 Pipeline (Most Common)
For SDR operations, simple scaling, ImageNet normalization:
```python
ep = ImagePreprocessor.from_scale_0_1(
    shape=(1, 3, 224, 224),
    input_dtype=torch.float16,
)
compile_specs = CoreMLBackend.generate_compile_specs(
    compute_precision=ct.precision.FLOAT16,  # Best ANE performance
    minimum_deployment_target=ct.target.iOS18,
)
```

### Pattern 2: fp16 I/O with fp32 Compute (Precision-Sensitive Operations)
For HDR10 PQ transfer function which has high-power exponents:
```python
ep = ImagePreprocessor.from_hdr10(
    shape=(1, 3, 1080, 1920),
    input_dtype=torch.float16,  # Memory efficient I/O
    output_dtype=torch.float16,
    bit_depth=10,
)
compile_specs = CoreMLBackend.generate_compile_specs(
    compute_precision=ct.precision.FLOAT32,  # Required for PQ accuracy
    minimum_deployment_target=ct.target.iOS18,
)
```

Reference correctness tests (comparison against colour-science) are in
extension/vision/test/test_image_processing.py. These CoreML tests focus on
verifying the CoreML delegate matches EP output and demonstrating deployment.
"""

import unittest

import coremltools as ct
import executorch.exir
import numpy as np
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)
from executorch.extension.vision.image_processing import (
    ColorGamut,
    ColorLayout,
    ImagePostprocessor,
    ImagePreprocessor,
    TransferFunction,
)

if IS_VALID_TEST_RUNTIME:
    from executorch.runtime import Runtime


class TestCoreMLImagePreprocessor(unittest.TestCase):
    """
    Tests for lowering ImagePreprocessor to CoreML.

    Each test method demonstrates the recommended deployment pattern for
    a specific preprocessor factory method.
    """

    # ==================== Helper Methods ====================

    def _lower_to_coreml(
        self,
        ep: torch.export.ExportedProgram,
        compute_precision: ct.precision = ct.precision.FLOAT16,
    ):
        """Lower ExportedProgram to CoreML-delegated ExecutorchProgram.

        Args:
            ep: The ExportedProgram to lower.
            compute_precision: CoreML compute precision.
                - ct.precision.FLOAT16: Best ANE performance (default)
                - ct.precision.FLOAT32: For precision-sensitive operations

        Returns:
            ExecutorchProgram ready for execution.
        """
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=compute_precision,
            minimum_deployment_target=ct.target.iOS18,
        )
        partitioner = CoreMLPartitioner(compile_specs=compile_specs)

        edge_program = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[partitioner]
        )

        # Verify all ops are delegated to CoreML
        for node in edge_program.exported_program().graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                is_delegate = "executorch_call_delegate" in target_str
                is_getitem = "getitem" in target_str
                self.assertTrue(
                    is_delegate or is_getitem,
                    f"Found non-delegated op: {node.target}",
                )

        return edge_program.to_executorch()

    def _run_coreml(self, executorch_program, inputs: torch.Tensor) -> np.ndarray:
        """Execute CoreML-delegated program and return output."""
        if not IS_VALID_TEST_RUNTIME:
            return None

        runtime = Runtime.get()
        program = runtime.load_program(executorch_program.buffer)
        method = program.load_method("forward")
        return method.execute([inputs])[0].numpy()

    def _generate_8bit_input(self, shape=(1, 3, 64, 64)) -> torch.Tensor:
        """Generate 8-bit test input (values 0-255)."""
        return torch.randint(0, 256, shape, dtype=torch.float16).float()

    def _generate_10bit_input(self, shape=(1, 3, 64, 64)) -> torch.Tensor:
        """Generate 10-bit test input (values 0-1023)."""
        return torch.randint(0, 1024, shape, dtype=torch.float16).float()

    def _generate_12bit_input(self, shape=(1, 3, 64, 64)) -> torch.Tensor:
        """Generate 12-bit test input (values 0-4095)."""
        return torch.randint(0, 4096, shape, dtype=torch.float16).float()

    # ==================== SDR Preprocessors (fp16 pipeline) ====================

    def test_from_scale_0_1(self):
        """
        Example: Scale 8-bit input to [0, 1] range.

        Use case: Generic preprocessing for models expecting normalized input.
        Precision: Full fp16 pipeline (best performance).
        """
        shape = (1, 3, 224, 224)

        # Create preprocessor with fp16 I/O
        ep = ImagePreprocessor.from_scale_0_1(
            shape=shape,
            input_dtype=torch.float16,
        )

        # Lower to CoreML with fp16 compute (best ANE performance)
        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        # Run and verify
        test_input = self._generate_8bit_input(shape).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        # Compare with EP output
        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    def test_from_scale_neg1_1(self):
        """
        Example: Scale 8-bit input to [-1, 1] range.

        Use case: Models expecting symmetric normalized input (e.g., some GANs).
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 224, 224)

        ep = ImagePreprocessor.from_scale_neg1_1(
            shape=shape,
            input_dtype=torch.float16,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        test_input = self._generate_8bit_input(shape).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    def test_from_imagenet(self):
        """
        Example: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

        Use case: Classification models trained on ImageNet (ResNet, EfficientNet, etc.).
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 224, 224)

        ep = ImagePreprocessor.from_imagenet(
            shape=shape,
            input_dtype=torch.float16,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        test_input = self._generate_8bit_input(shape).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    def test_from_sdr_to_linear(self):
        """
        Example: Convert SDR (sRGB gamma) to linear light.

        Use case: HDR processing pipelines, tone mapping, color grading.
        Precision: Full fp16 pipeline (sRGB gamma is well-behaved).
        """
        shape = (1, 3, 224, 224)

        ep = ImagePreprocessor.from_sdr(
            shape=shape,
            input_dtype=torch.float16,
            output_dtype=torch.float16,
            normalize_to_linear=True,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        test_input = self._generate_8bit_input(shape).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    # ==================== HDR Preprocessors ====================

    def test_from_hlg(self):
        """
        Example: Convert HLG (Hybrid Log-Gamma) to linear light.

        Use case: Processing HLG HDR content (common in broadcast).
        Precision: Full fp16 pipeline (HLG is well-behaved in fp16).
        """
        shape = (1, 3, 1080, 1920)

        ep = ImagePreprocessor.from_hlg(
            shape=shape,
            input_dtype=torch.float16,
            output_dtype=torch.float16,
            bit_depth=10,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        test_input = self._generate_10bit_input(shape).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.05, atol=0.05)

    def test_from_hdr10(self):
        """
        Example: Convert HDR10 (PQ/ST.2084) to linear light.

        Use case: Processing HDR10 content (streaming, UHD Blu-ray).

        IMPORTANT: HDR10 uses the PQ transfer function which has high-power
        exponents (m2=78.84). This causes significant precision loss in fp16.
        Use fp32 compute precision for accurate results.

        Precision: fp16 I/O with fp32 compute (required for PQ accuracy).
        """
        shape = (1, 3, 1080, 1920)

        # Create with fp16 I/O for memory efficiency
        ep = ImagePreprocessor.from_hdr10(
            shape=shape,
            input_dtype=torch.float16,
            output_dtype=torch.float16,
            bit_depth=10,
        )

        # Use fp32 compute precision for PQ accuracy
        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT32
        )

        test_input = self._generate_10bit_input(shape).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        # For comparison, create fp32 EP to match fp32 compute precision
        ep_fp32 = ImagePreprocessor.from_hdr10(
            shape=shape,
            input_dtype=torch.float32,
            output_dtype=torch.float32,
            bit_depth=10,
        )
        ep_out = ep_fp32.module()(test_input.to(torch.float32)).detach().numpy()
        ep_out = ep_out.astype(np.float16)  # Cast to match CoreML output dtype

        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    # ==================== Color Layout Conversions ====================

    def test_bgr_to_rgb(self):
        """
        Example: Convert BGR input to RGB output.

        Use case: OpenCV uses BGR format; convert for RGB-expecting models.
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 224, 224)

        ep = ImagePreprocessor.from_scale_0_1(
            shape=shape,
            input_dtype=torch.float16,
            input_color=ColorLayout.BGR,
            output_color=ColorLayout.RGB,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        test_input = self._generate_8bit_input(shape).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    def test_rgb_to_grayscale(self):
        """
        Example: Convert RGB to grayscale.

        Use case: Models expecting single-channel input.
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 224, 224)

        ep = ImagePreprocessor.from_scale_0_1(
            shape=shape,
            input_dtype=torch.float16,
            input_color=ColorLayout.RGB,
            output_color=ColorLayout.GRAYSCALE,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        test_input = self._generate_8bit_input(shape).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    # ==================== Gamut Conversion ====================

    def test_bt2020_to_bt709(self):
        """
        Example: Convert BT.2020 (wide gamut) to BT.709 (SDR gamut).

        Use case: HDR to SDR conversion, displaying HDR content on SDR screens.
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 1080, 1920)

        model = ImagePreprocessor(
            bit_depth=10,
            input_transfer=TransferFunction.LINEAR,
            output_transfer=TransferFunction.LINEAR,
            input_gamut=ColorGamut.BT2020,
            output_gamut=ColorGamut.BT709,
        )
        model.eval()
        test_input = self._generate_10bit_input(shape).to(torch.float16)
        ep = torch.export.export(model, (test_input,), strict=True)

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)


class TestCoreMLImagePostprocessor(unittest.TestCase):
    """
    Tests for lowering ImagePostprocessor to CoreML.

    Each test method demonstrates the recommended deployment pattern for
    a specific postprocessor factory method.
    """

    # ==================== Helper Methods ====================

    def _lower_to_coreml(
        self,
        ep: torch.export.ExportedProgram,
        compute_precision: ct.precision = ct.precision.FLOAT16,
    ):
        """Lower ExportedProgram to CoreML-delegated ExecutorchProgram."""
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=compute_precision,
            minimum_deployment_target=ct.target.iOS18,
        )
        partitioner = CoreMLPartitioner(compile_specs=compile_specs)

        edge_program = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[partitioner]
        )

        for node in edge_program.exported_program().graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                is_delegate = "executorch_call_delegate" in target_str
                is_getitem = "getitem" in target_str
                self.assertTrue(
                    is_delegate or is_getitem,
                    f"Found non-delegated op: {node.target}",
                )

        return edge_program.to_executorch()

    def _run_coreml(self, executorch_program, inputs: torch.Tensor) -> np.ndarray:
        """Execute CoreML-delegated program and return output."""
        if not IS_VALID_TEST_RUNTIME:
            return None

        runtime = Runtime.get()
        program = runtime.load_program(executorch_program.buffer)
        method = program.load_method("forward")
        return method.execute([inputs])[0].numpy()

    # ==================== SDR Postprocessors (fp16 pipeline) ====================

    def test_from_scale_0_1(self):
        """
        Example: Convert [0, 1] normalized output to 8-bit [0, 255].

        Use case: Converting model output back to displayable image.
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 224, 224)

        ep = ImagePostprocessor.from_scale_0_1(
            shape=shape,
            input_dtype=torch.float16,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        # Input is [0, 1] normalized
        test_input = torch.rand(shape, dtype=torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    def test_from_scale_neg1_1(self):
        """
        Example: Convert [-1, 1] normalized output to 8-bit [0, 255].

        Use case: GAN output conversion.
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 224, 224)

        ep = ImagePostprocessor.from_scale_neg1_1(
            shape=shape,
            input_dtype=torch.float16,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        # Input is [-1, 1] normalized
        test_input = torch.rand(shape, dtype=torch.float16) * 2 - 1
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    def test_from_imagenet(self):
        """
        Example: Reverse ImageNet normalization.

        Use case: Visualizing model intermediate outputs or reconstructions.
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 224, 224)

        ep = ImagePostprocessor.from_imagenet(
            shape=shape,
            input_dtype=torch.float16,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        # Input is ImageNet-normalized
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        test_input = ((torch.rand(shape) - mean) / std).to(torch.float16)

        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.02, atol=0.1)

    def test_from_linear_to_srgb(self):
        """
        Example: Convert linear light to sRGB gamma.

        Use case: Displaying linear light processing results.
        Precision: Full fp16 pipeline.
        """
        shape = (1, 3, 224, 224)

        ep = ImagePostprocessor.from_linear_to_srgb(
            shape=shape,
            input_dtype=torch.float16,
            output_dtype=torch.float16,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        # Input is linear light [0, 1]
        test_input = torch.rand(shape, dtype=torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.01, atol=0.01)

    # ==================== HDR Postprocessors ====================

    def test_from_linear_to_hlg(self):
        """
        Example: Convert linear light to HLG.

        Use case: Encoding processed content for HLG HDR display.
        Precision: Full fp16 pipeline (HLG is well-behaved).
        """
        shape = (1, 3, 1080, 1920)

        ep = ImagePostprocessor.from_linear_to_hlg(
            shape=shape,
            input_dtype=torch.float16,
            output_dtype=torch.float16,
        )

        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT16
        )

        # Input is linear light [0, 1]
        test_input = torch.rand(shape, dtype=torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        ep_out = ep.module()(test_input).detach().numpy()
        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.05, atol=0.05)

    def test_from_linear_to_hdr10(self):
        """
        Example: Convert linear light to HDR10 (PQ/ST.2084).

        Use case: Encoding processed content for HDR10 display.

        IMPORTANT: HDR10 uses the PQ transfer function which has high-power
        exponents (m2=78.84). Use fp32 compute precision for accurate results.

        Precision: fp16 I/O with fp32 compute (required for PQ accuracy).
        """
        shape = (1, 3, 1080, 1920)

        # Create with fp16 I/O for memory efficiency
        ep = ImagePostprocessor.from_linear_to_hdr10(
            shape=shape,
            input_dtype=torch.float16,
            output_dtype=torch.float16,
        )

        # Use fp32 compute precision for PQ accuracy
        executorch_program = self._lower_to_coreml(
            ep, compute_precision=ct.precision.FLOAT32
        )

        # Input is linear light [0, 1], avoid very small values
        test_input = (torch.rand(shape) * 0.99 + 0.01).to(torch.float16)
        coreml_out = self._run_coreml(executorch_program, test_input)

        if coreml_out is None:
            self.skipTest("CoreML runtime not available")

        # For comparison, create fp32 EP to match fp32 compute precision
        ep_fp32 = ImagePostprocessor.from_linear_to_hdr10(
            shape=shape,
            input_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        ep_out = ep_fp32.module()(test_input.to(torch.float32)).detach().numpy()
        ep_out = ep_out.astype(np.float16)  # Cast to match CoreML output dtype

        np.testing.assert_allclose(coreml_out, ep_out, rtol=0.05, atol=35.0)


if __name__ == "__main__":
    unittest.main()
