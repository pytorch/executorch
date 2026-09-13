# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import Counter
from typing import Tuple

import torch
from executorch.backends.arm._passes.fuse_quantized_activation_pass import (
    FuseQuantizedActivationPass,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.exir.dialects._ops import ops as exir_ops

input_t = Tuple[torch.Tensor]


class ConvRelu(torch.nn.Module):
    """Conv2d followed by ReLU — existing fuseable behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.relu = torch.nn.ReLU()

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def _activation_and_quant(
    target,
    activation_args,
    scale: float = 0.1,
    zp: int = 0,
    qmin: int = -128,
    qmax: int = 127,
    dtype: torch.dtype = torch.int8,
):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    conv = graph.call_function(exir_ops.edge.aten.convolution.default, (x,))
    activation = graph.call_function(target, (conv, *activation_args))
    quant = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        (activation, scale, zp, qmin, qmax, dtype),
    )
    return activation, quant


def _activation_and_tensor_qparam_quant(
    target,
    activation_args,
    qmin: int = -128,
    qmax: int = 127,
    dtype: torch.dtype = torch.int8,
):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    scale = graph.placeholder("scale")
    zp = graph.placeholder("zp")
    conv = graph.call_function(exir_ops.edge.aten.convolution.default, (x,))
    activation = graph.call_function(target, (conv, *activation_args))
    quant = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        (activation, scale, zp, qmin, qmax, dtype),
    )
    return activation, quant


def _activation_quant_graph(
    target,
    activation_args,
    scale: float = 0.1,
    zp: int = 0,
    qmin: int = -128,
    qmax: int = 127,
    dtype: torch.dtype = torch.int8,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 4))
    weight = graph.placeholder("weight")
    weight.meta["val"] = torch.empty((4, 4))
    bias = graph.placeholder("bias")
    bias.meta["val"] = torch.empty(4)
    linear = graph.call_function(exir_ops.edge.aten.linear.default, (x, weight, bias))
    linear.meta["val"] = torch.empty((2, 4))
    activation = graph.call_function(target, (linear, *activation_args))
    activation.meta["val"] = torch.empty((2, 4))
    quant = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        (activation, scale, zp, qmin, qmax, dtype),
    )
    quant.meta["val"] = torch.empty((2, 4), dtype=dtype)
    graph.output(quant)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _assert_fusion_preserves_output(graph_module: torch.fx.GraphModule) -> None:
    original = copy.deepcopy(graph_module)
    result = FuseQuantizedActivationPass().call(graph_module)
    assert result.modified

    inputs = (
        torch.tensor(
            [[-5.0, -0.06, 0.0, 0.04], [1.2, 12.7, 13.0, 20.0]],
        ),
        torch.eye(4),
        torch.zeros(4),
    )
    with torch.no_grad():
        expected = original(*inputs)
        actual = result.graph_module(*inputs)

    torch.testing.assert_close(actual, expected)


def test_fuse_relu_after_conv_quantized() -> None:
    """Existing behavior: ReLU after conv is fused in quantized graph."""
    module = ConvRelu()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=True,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_relu_default",
        ],
        pass_list=[FuseQuantizedActivationPass],
    )
    pipeline.run()


def test_relu_fusion_rewrites_qmin_to_zero_point() -> None:
    activation, quant = _activation_and_quant(
        exir_ops.edge.aten.relu.default,
        (),
    )

    assert FuseQuantizedActivationPass._is_fuseable_quantized_activation(activation)
    FuseQuantizedActivationPass._set_quant_qmin_to_zp(quant)

    assert quant.args[3] == 0
    assert quant.args[4] == 127


def test_relu_fusion_preserves_quantized_values() -> None:
    graph_module = _activation_quant_graph(
        exir_ops.edge.aten.relu.default,
        (),
        zp=-3,
    )

    _assert_fusion_preserves_output(graph_module)
    counts = Counter(
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    )
    assert counts[exir_ops.edge.aten.relu.default] == 0


def test_relu_with_tensor_qparams_is_not_fuseable() -> None:
    activation, quant = _activation_and_tensor_qparam_quant(
        exir_ops.edge.aten.relu.default,
        (),
    )

    assert not FuseQuantizedActivationPass._is_fuseable_quantized_activation(activation)
    assert quant.args[3] == -128
    assert quant.args[4] == 127


def test_hardtanh_with_non_saturating_max_is_not_fuseable() -> None:
    activation, quant = _activation_and_quant(
        exir_ops.edge.aten.hardtanh.default,
        (0.0, 6.0),
    )

    assert not FuseQuantizedActivationPass._is_fuseable_quantized_activation(activation)
    assert quant.args[3] == -128
    assert quant.args[4] == 127


def test_hardtanh_with_saturating_max_rewrites_qmin_to_zero_point() -> None:
    activation, quant = _activation_and_quant(
        exir_ops.edge.aten.hardtanh.default,
        (0.0, 13.0),
    )

    assert FuseQuantizedActivationPass._is_fuseable_quantized_activation(activation)
    FuseQuantizedActivationPass._set_quant_qmin_to_zp(quant)

    assert quant.args[3] == 0
    assert quant.args[4] == 127


def test_saturating_hardtanh_fusion_preserves_quantized_values() -> None:
    graph_module = _activation_quant_graph(
        exir_ops.edge.aten.hardtanh.default,
        (0.0, 13.0),
    )

    _assert_fusion_preserves_output(graph_module)
    counts = Counter(
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    )
    assert counts[exir_ops.edge.aten.hardtanh.default] == 0
