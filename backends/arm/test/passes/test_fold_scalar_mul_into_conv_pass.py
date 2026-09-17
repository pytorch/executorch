# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import Tuple

import torch
from executorch.backends.arm._passes.arm_pass_manager import (
    _ExportedProgramGraphPassAdapter,
)
from executorch.backends.arm._passes.fold_scalar_mul_into_conv_pass import (
    FoldScalarMulIntoConvPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


def _run_pass(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]):
    exported_program = torch.export.export(module.eval(), inputs, strict=True)
    edge_program = to_edge(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_exported_program = edge_program.exported_program()
    transformed = edge_program.transform(
        [
            _ExportedProgramGraphPassAdapter(
                FoldScalarMulIntoConvPass(edge_exported_program)
            )
        ]
    )
    return transformed.exported_program()


def _op_counts(exported_program) -> Counter:
    return Counter(
        node.target
        for node in exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    )


def _placeholder_names(exported_program) -> list[str]:
    return [
        node.name
        for node in exported_program.graph_module.graph.nodes
        if node.op == "placeholder"
    ]


def _unused_placeholder_names(exported_program) -> list[str]:
    return [
        node.name
        for node in exported_program.graph_module.graph.nodes
        if node.op == "placeholder" and len(node.users) == 0
    ]


class ConvMulChannelScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.scale = torch.nn.Parameter(
            torch.tensor([1.25, -0.5, 2.0, 0.75]).reshape(1, 4, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * self.scale


class ConvMulScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.25 * self.conv(x)


class ConvMulMultipleUsers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y * 0.25 + y


class ConvMulWidthScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.scale = torch.nn.Parameter(torch.tensor([1.25, -0.5, 2.0, 0.75]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * self.scale


def test_fold_aten_conv2d_channel_scale_mul_before_edge() -> None:
    module = ConvMulChannelScale().eval()
    inputs = (torch.randn(2, 3, 8, 8),)
    exported_program = torch.export.export(module, inputs, strict=True)

    pass_result = _ExportedProgramGraphPassAdapter(
        FoldScalarMulIntoConvPass(exported_program, tfa_pass=True)
    )(exported_program)
    assert pass_result is not None
    counts = Counter(
        node.target
        for node in pass_result.exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    )

    assert pass_result.modified
    assert counts[torch.ops.aten.conv2d.default] == 1
    assert counts[torch.ops.aten.mul.Tensor] == 0
    torch.testing.assert_close(module(*inputs), exported_program.module()(*inputs))


def test_fold_get_attr_conv2d_channel_scale_mul_as_tfa_pass() -> None:
    module = ConvMulChannelScale().eval()
    inputs = (torch.randn(2, 3, 8, 8),)
    expected = module(*inputs)
    graph_module = torch.export.export(module, inputs, strict=True).module()

    pass_result = FoldScalarMulIntoConvPass(tfa_pass=True)(graph_module)
    assert pass_result is not None
    counts = Counter(
        node.target
        for node in pass_result.graph_module.graph.nodes
        if node.op == "call_function"
    )

    assert pass_result.modified
    assert counts[torch.ops.aten.conv2d.default] == 1
    assert counts[torch.ops.aten.mul.Tensor] == 0
    torch.testing.assert_close(expected, pass_result.graph_module(*inputs))


def test_fold_channel_scale_mul_into_conv() -> None:
    module = ConvMulChannelScale().eval()
    inputs = (torch.randn(2, 3, 8, 8),)

    transformed = _run_pass(module, inputs)
    counts = _op_counts(transformed)

    assert counts[exir_ops.edge.aten.convolution.default] == 1
    assert counts[exir_ops.edge.aten.mul.Tensor] == 0
    assert _unused_placeholder_names(transformed) == []
    assert [
        spec.arg.name for spec in transformed.graph_signature.input_specs
    ] == _placeholder_names(transformed)
    torch.testing.assert_close(module(*inputs), transformed.module()(*inputs))


def test_fold_scalar_mul_into_conv() -> None:
    module = ConvMulScalar().eval()
    inputs = (torch.randn(2, 3, 8, 8),)

    transformed = _run_pass(module, inputs)
    counts = _op_counts(transformed)

    assert counts[exir_ops.edge.aten.convolution.default] == 1
    assert counts[exir_ops.edge.aten.mul.Tensor] == 0
    assert counts[exir_ops.edge.aten.mul.Scalar] == 0
    torch.testing.assert_close(module(*inputs), transformed.module()(*inputs))


def test_does_not_fold_1d_width_scale_as_channel_scale() -> None:
    module = ConvMulWidthScale().eval()
    inputs = (torch.randn(2, 3, 4, 4),)

    transformed = _run_pass(module, inputs)
    counts = _op_counts(transformed)

    assert counts[exir_ops.edge.aten.convolution.default] == 1
    assert counts[exir_ops.edge.aten.mul.Tensor] == 1
    torch.testing.assert_close(module(*inputs), transformed.module()(*inputs))


def test_does_not_fold_when_conv_has_multiple_users() -> None:
    module = ConvMulMultipleUsers().eval()
    inputs = (torch.randn(2, 3, 8, 8),)

    transformed = _run_pass(module, inputs)
    counts = _op_counts(transformed)

    assert counts[exir_ops.edge.aten.convolution.default] == 1
    mul_tensor_count = counts[exir_ops.edge.aten.mul.Tensor]
    mul_scalar_count = counts[exir_ops.edge.aten.mul.Scalar]
    mul_count = mul_tensor_count + mul_scalar_count
    assert mul_count == 1
    torch.testing.assert_close(module(*inputs), transformed.module()(*inputs))
