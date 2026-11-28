# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401

from executorch.backends.arm.quantizer.arm_quantizer_utils import mark_node_as_annotated
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


class QuantizedConvAdd(torch.nn.Module):
    """Keep convolution quantization independent of calibration."""

    def __init__(self, path: str, weight: int, conv_scale: float):
        super().__init__()
        self.path = path
        self.conv_scale = conv_scale
        self.register_buffer("weight", torch.tensor([[[[weight]]]], dtype=torch.int8))
        self.register_buffer("bias", torch.tensor([0], dtype=torch.int32))
        self.register_buffer("scales", torch.tensor([1.0]))
        self.register_buffer("zeroes", torch.tensor([0], dtype=torch.int64))

    @staticmethod
    def quantize(value: torch.Tensor, scale: float) -> torch.Tensor:
        """Apply the fixed INT16 contract without calibration."""
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            value, scale, 0, -32768, 32767, torch.int16
        )

    @staticmethod
    def dequantize(value: torch.Tensor, scale: float) -> torch.Tensor:
        """Restore float values on the selected quantization grid."""
        return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            value, scale, 0, -32768, 32767, torch.int16
        )

    def forward(self, image: torch.Tensor, residual: torch.Tensor):
        """Expose the narrowed convolution to one or multiple consumers."""
        image = self.dequantize(self.quantize(image, 1.0), 1.0)
        residual = self.dequantize(self.quantize(residual, 1.0), 1.0)
        weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.weight, self.scales, self.zeroes, 0, -127, 127, torch.int8
        )
        bias = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.bias, self.scales, self.zeroes, 0, -2147483647, 2147483647, torch.int32
        )
        conv = torch.nn.functional.conv2d(image, weight, bias)
        conv = self.dequantize(self.quantize(conv, self.conv_scale), self.conv_scale)
        if self.path == "permuted":
            conv = self.dequantize(
                self.quantize(conv.permute(0, 1, 3, 2), self.conv_scale),
                self.conv_scale,
            )
            residual = self.dequantize(
                self.quantize(residual.permute(0, 1, 3, 2), 1.0), 1.0
            )
        result = self.dequantize(self.quantize(conv + residual, 1.0), 1.0)
        if self.path == "multiple":
            relu = self.dequantize(
                self.quantize(torch.relu(conv), self.conv_scale), self.conv_scale
            )
            return result, relu
        return (result,)


@pytest.mark.parametrize("path", ["direct", "permuted", "multiple"])
@pytest.mark.parametrize(
    "image,residual,weight,conv_scale,expected",
    [
        (20000, -30000, 2, 1.0, 2767),
        (-20000, 30000, 2, 1.0, -2768),
        (1, 0, 3, 2.0, 4),
    ],
    ids=["positive-saturation", "negative-saturation", "intermediate-rounding"],
)
def test_a16_conv_rescale_boundary_matches_native_tosa(
    tmp_path: Path,
    path: str,
    image: int,
    residual: int,
    weight: int,
    conv_scale: float,
    expected: int,
) -> None:
    """Compare saturation and rounding with the native TOSA model."""
    reference_model = pytest.importorskip("tosa_reference_model")
    model = QuantizedConvAdd(path, weight, conv_scale)
    inputs = (
        torch.full((1, 1, 2, 3), image, dtype=torch.float32),
        torch.full((1, 1, 2, 3), residual, dtype=torch.float32),
    )
    program = torch.export.export(model, inputs)
    # The graph has explicit quantization already; mark those operations just
    # as the Arm quantizer does so the partitioner accepts the INT-only region.
    for node in program.graph.nodes:
        if node.target in (
            torch.ops.aten.conv2d.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.permute.default,
            torch.ops.aten.relu.default,
        ):
            mark_node_as_annotated(node)
    cpu = program.module()(*inputs)
    assert torch.all(cpu[0] == expected)
    spec = TosaCompileSpec("TOSA-1.0+INT+int16").dump_intermediate_artifacts_to(
        str(tmp_path)
    )
    edge = to_edge_transform_and_lower(
        program,
        partitioner=[TOSAPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    delegates = [
        node
        for node in edge.exported_program().graph.nodes
        if node.op == "call_function" and "delegate" in str(node.target)
    ]
    assert len(delegates) == 1
    fallback = [
        node.target
        for node in edge.exported_program().graph.nodes
        if node.op == "call_function"
        and node not in delegates
        and node.target != operator.getitem
        and "quantized_decomposed" not in str(node.target)
    ]
    assert fallback == []
    (tosa_path,) = tuple(tmp_path.glob("*.tosa"))
    outputs, status = reference_model.run(
        tosa_path.read_bytes(),
        [value.numpy().astype(np.int16) for value in inputs],
        initialize_variable_tensor_from_numpy=True,
        verbosity="NONE",
    )
    assert status == reference_model.GraphStatus.TOSA_VALID
    assert len(outputs) == len(cpu)
    for index, (expected_output, actual_output) in enumerate(zip(cpu, outputs)):
        output_scale = conv_scale if path == "multiple" and index == 1 else 1.0
        expected_codes = model.quantize(expected_output, output_scale)
        np.testing.assert_array_equal(actual_output, expected_codes.numpy())
