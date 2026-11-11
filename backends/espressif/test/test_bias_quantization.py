# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
import pytest
import torch
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from executorch.backends.espressif.reference import ProgramEmulator
from executorch.backends.espressif.serialization.xts3 import (
    build_graph,
    ExportedTensorResolver,
    parse_graph,
    serialize_graph,
    XTS3OpCode,
)
from executorch.backends.transforms.utils import get_param_tensor
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer import ComposableQuantizer


def _quantize(model, inputs, quantizer, *, qat=False):
    captured = torch.export.export(model, inputs, strict=True).module()
    prepare = prepare_qat_pt2e if qat else prepare_pt2e
    prepared = prepare(captured, quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared)
    return converted, torch.export.export(converted, inputs, strict=True)


@pytest.fixture(params=["linear", "conv2d"])
def weighted_case(request):
    if request.param == "linear":
        model = torch.nn.Linear(2, 3)
        shape = (2, 2)
    else:
        model = torch.nn.Conv2d(2, 3, 1)
        shape = (1, 2, 2, 2)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[0.2, -0.1], [0.03, 0.05], [-0.4, 0.1]]).reshape_as(
                model.weight
            )
        )
        model.bias.copy_(torch.tensor([0.125, -0.3, 0.6]))
    return model.eval(), (torch.linspace(-2, 3, int(np.prod(shape))).reshape(shape),)


def _weighted_node(program):
    return next(
        node
        for node in program.graph.nodes
        if node.target in (torch.ops.aten.linear.default, torch.ops.aten.conv2d.default)
    )


def _stored_bias(graph):
    operation = next(
        op
        for op in graph.operations
        if op.opcode in (XTS3OpCode.LINEAR, XTS3OpCode.CONV2D)
    )
    bias = graph.tensors[operation.inputs[2]]
    return np.frombuffer(
        graph.constant_data[bias.data_offset : bias.data_offset + bias.data_size],
        dtype=np.int32,
    )


@pytest.mark.parametrize(
    "activation", [torch.nn.Identity, torch.nn.ReLU, torch.nn.ReLU6]
)
def test_bias_quantization_and_delegation(weighted_case, activation):
    op, inputs = weighted_case
    model = torch.nn.Sequential(op, activation()).eval()
    converted, program = _quantize(model, inputs, EspNNQuantizer())
    node = _weighted_node(program)
    resolver = ExportedTensorResolver(program)
    bias = node.args[2]
    assert bias.target == torch.ops.quantized_decomposed.dequantize_per_channel.default
    assert bias.args[3] == 0
    assert bias.args[6] == torch.int32
    input_scale = node.args[0].args[1]
    weight_scales = resolver.resolve(node.args[1].args[1])
    torch.testing.assert_close(
        resolver.resolve(bias.args[1]), weight_scales * input_scale, atol=0, rtol=0
    )
    assert torch.all(resolver.resolve(bias.args[2]) == 0)

    graph = parse_graph(serialize_graph(build_graph(program)))
    np.testing.assert_array_equal(
        _stored_bias(graph), resolver.resolve(bias.args[0]).numpy()
    )
    edge, report = to_espnn_edge(program, allow_cpu_fallback=False)
    assert report["partitions"] == 1
    assert not report["cpu_ops"]
    output_scale = graph.tensors[graph.outputs[0]].quant_scale
    torch.testing.assert_close(
        ProgramEmulator(edge.exported_program())(*inputs),
        converted(*inputs),
        atol=output_scale,
        rtol=0,
    )
    assert edge.to_executorch().buffer


def test_delegate_accepts_float_bias(weighted_case):
    model, inputs = weighted_case
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True)
    )
    converted, program = _quantize(model, inputs, quantizer)
    bias = _weighted_node(program).args[2]
    assert bias.op == "placeholder"
    assert ExportedTensorResolver(program).resolve(bias).dtype == torch.float32
    graph = build_graph(program)
    edge, _ = to_espnn_edge(program, allow_cpu_fallback=False)
    torch.testing.assert_close(
        ProgramEmulator(edge.exported_program())(*inputs),
        converted(*inputs),
        atol=graph.tensors[graph.outputs[0]].quant_scale,
        rtol=0,
    )
    assert edge.to_executorch().buffer


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("activation", [torch.nn.Identity, torch.nn.ReLU6])
def test_bias_annotation_matches_weight_scheme_and_qat(
    weighted_case, per_channel, activation
):
    model, inputs = weighted_case
    _, program = _quantize(
        torch.nn.Sequential(model, activation()),
        inputs,
        EspNNQuantizer(per_channel=per_channel, is_qat=True),
        qat=True,
    )
    bias = _weighted_node(program).args[2]
    target = (
        torch.ops.quantized_decomposed.dequantize_per_channel.default
        if per_channel
        else torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert bias.target == target
    assert torch.int32 in bias.args


def test_bias_annotation_respects_filter(weighted_case):
    model, inputs = weighted_case
    quantizer = EspNNQuantizer().set_filter_function(lambda node: False)
    converted, program = _quantize(model, inputs, quantizer)
    assert _weighted_node(program).args[2].op == "placeholder"
    torch.testing.assert_close(converted(*inputs), model(*inputs), atol=0, rtol=0)


@pytest.mark.parametrize("exclusion", ["none", "filter", "config"])
def test_bias_annotation_preserves_other_quantizers(weighted_case, exclusion):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x):
            return self.op(x), self.op.bias

    op, inputs = weighted_case
    model = Model(op).eval()
    config = get_symmetric_quantization_config(is_per_channel=True)
    reference, _ = _quantize(model, inputs, XNNPACKQuantizer().set_global(config))
    esp = EspNNQuantizer()
    if exclusion == "filter":
        esp.set_filter_function(lambda node: False)
    elif exclusion == "config":
        esp.set_global(None)
    converted, program = _quantize(
        model,
        inputs,
        ComposableQuantizer([XNNPACKQuantizer().set_global(config), esp]),
    )
    bias = _weighted_node(program).args[2]
    assert bias.op == "placeholder"
    assert ExportedTensorResolver(program).resolve(bias).dtype == torch.float32
    assert all(
        node.target != torch.ops.aten.clone.default for node in program.graph.nodes
    )
    torch.testing.assert_close(converted(*inputs), reference(*inputs), atol=0, rtol=0)


@pytest.mark.parametrize("qat", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
def test_bias_int32_boundaries(weighted_case, qat, per_channel):
    model, (input_tensor,) = weighted_case
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([127.0, 128.0, -128.0]))
    inputs = (torch.zeros_like(input_tensor),)
    converted, program = _quantize(
        model,
        inputs,
        EspNNQuantizer(per_channel=per_channel, is_qat=qat),
        qat=qat,
    )
    node = _weighted_node(program)
    bias = node.args[2]
    torch.testing.assert_close(
        ExportedTensorResolver(program).resolve(bias.args[0]),
        torch.tensor([2130706432, 2147483520, -2147483648], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    output_scale = next(iter(node.users)).args[1]
    torch.testing.assert_close(
        converted(*inputs), model(*inputs), atol=output_scale, rtol=0
    )
    if per_channel:
        edge, _ = to_espnn_edge(program, allow_cpu_fallback=False)
        torch.testing.assert_close(
            ProgramEmulator(edge.exported_program())(*inputs),
            converted(*inputs),
            atol=output_scale,
            rtol=0,
        )
        assert edge.to_executorch().buffer


@pytest.mark.parametrize("qat", [False, True])
def test_shared_runtime_bias_stays_float_without_copies(weighted_case, qat):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.weight = op.weight
            self.other_weight = torch.nn.Parameter(op.weight.detach().clone())
            self.operator = (
                torch.nn.functional.linear
                if isinstance(op, torch.nn.Linear)
                else torch.nn.functional.conv2d
            )

        def forward(self, x, y, bias):
            return (
                self.operator(x, self.weight, bias),
                self.operator(y, self.other_weight, bias),
                bias,
            )

    op, (input_tensor,) = weighted_case
    inputs = (input_tensor, input_tensor * 8, op.bias.detach().clone())
    converted, program = _quantize(
        Model(op).eval(), inputs, EspNNQuantizer(is_qat=qat), qat=qat
    )
    nodes = [
        node
        for node in program.graph.nodes
        if node.target in (torch.ops.aten.linear.default, torch.ops.aten.conv2d.default)
    ]
    assert len(nodes) == 2
    for node in nodes:
        bias = node.args[2]
        assert bias.op == "placeholder"
        assert bias.meta["val"].dtype == torch.float32
    edge, _ = to_espnn_edge(program, use_operator_library=True)
    actual = ProgramEmulator(edge.exported_program())(*inputs)
    torch.testing.assert_close(actual, converted(*inputs), atol=0, rtol=0)
    torch.testing.assert_close(actual[2], inputs[2], atol=0, rtol=0)
    runtime = edge.to_executorch()
    assert all(
        "clone" not in op.name
        for op in runtime.executorch_program.execution_plan[0].operators
    )


@pytest.mark.parametrize("qat", [False, True])
def test_shared_bias_uses_each_calls_input_scale(weighted_case, qat):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.first = op
            self.second = copy.deepcopy(op)
            self.second.bias = op.bias

        def forward(self, x, y):
            return self.first(x), self.second(y)

    op, (inputs,) = weighted_case
    _, program = _quantize(
        Model(op), (inputs, inputs * 8), EspNNQuantizer(is_qat=qat), qat=qat
    )
    resolver = ExportedTensorResolver(program)
    bias_sources = set()
    for node in program.graph.nodes:
        assert node.target != torch.ops.aten.clone.default
        if node.target not in (
            torch.ops.aten.linear.default,
            torch.ops.aten.conv2d.default,
        ):
            continue
        bias = node.args[2]
        bias_sources.add(bias.args[0])
        torch.testing.assert_close(
            resolver.resolve(bias.args[1]),
            resolver.resolve(node.args[1].args[1]) * node.args[0].args[1],
            atol=0,
            rtol=0,
        )
    assert len(bias_sources) == 2
    edge, _ = to_espnn_edge(program, allow_cpu_fallback=False)
    assert edge.to_executorch().buffer


@pytest.mark.parametrize("use_operator_library", [False, True])
def test_bias_rounding_is_visible_before_lowering(use_operator_library):
    model = torch.nn.Linear(2, 1).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.bias.fill_(0.1)
    inputs = (torch.tensor([[0.0, 100000.0], [0.0, -100000.0]]),)
    converted, program = _quantize(model, inputs, EspNNQuantizer())
    torch.testing.assert_close(model(*inputs), torch.full((2, 1), 0.1), atol=0, rtol=0)
    torch.testing.assert_close(converted(*inputs), torch.zeros(2, 1), atol=0, rtol=0)
    edge, _ = to_espnn_edge(
        program, allow_cpu_fallback=False, use_operator_library=use_operator_library
    )
    torch.testing.assert_close(
        ProgramEmulator(edge.exported_program())(*inputs),
        converted(*inputs),
        atol=0,
        rtol=0,
    )


def test_prepared_bias_is_not_rounded_again(weighted_case):
    model, inputs = weighted_case
    with torch.no_grad():
        model.bias.copy_(torch.tensor([500.0, 100.0, 0.1]))
    _, program = _quantize(model, inputs, EspNNQuantizer())
    node = _weighted_node(program)
    resolver = ExportedTensorResolver(program)
    bias = node.args[2]
    expected = resolver.resolve(bias.args[0])
    recomputed = torch.round(
        resolver.resolve(bias).to(torch.float64)
        / (
            node.args[0].args[1]
            * resolver.resolve(node.args[1].args[1]).to(torch.float64)
        )
    ).to(torch.int32)
    assert torch.any(expected != recomputed)
    np.testing.assert_array_equal(_stored_bias(build_graph(program)), expected.numpy())
    if node.target == torch.ops.aten.linear.default:
        edge, _ = to_espnn_edge(
            program, allow_cpu_fallback=False, use_operator_library=True
        )
        result = edge.exported_program()
        lowered = next(
            n
            for n in result.graph.nodes
            if n.target == exir_ops.edge.esp_nn.quantized_linear_int8.default
        )
        torch.testing.assert_close(
            get_param_tensor(result, lowered.args[2]), expected, atol=0, rtol=0
        )


def test_other_bias_scales_keep_existing_conversion(weighted_case):
    model, inputs = weighted_case
    _, program = _quantize(model, inputs, EspNNQuantizer())
    node = _weighted_node(program)
    resolver = ExportedTensorResolver(program)
    bias = node.args[2]
    resolver.resolve(bias.args[1]).mul_(2)
    expected = torch.round(
        resolver.resolve(bias).to(torch.float64)
        / (
            node.args[0].args[1]
            * resolver.resolve(node.args[1].args[1]).to(torch.float64)
        )
    ).to(torch.int32)
    np.testing.assert_array_equal(_stored_bias(build_graph(program)), expected.numpy())


@pytest.mark.parametrize("use_operator_library", [False, True])
def test_float_bias_consumer_is_preserved(use_operator_library):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 3)

        def forward(self, x):
            return self.linear(x), self.linear.bias

    model = Model().eval()
    inputs = (torch.tensor([[-2.0, 3.0]]),)
    converted, program = _quantize(model, inputs, EspNNQuantizer())
    torch.testing.assert_close(converted(*inputs)[1], model.linear.bias, atol=0, rtol=0)
    edge, _ = to_espnn_edge(
        program, allow_cpu_fallback=False, use_operator_library=use_operator_library
    )
    torch.testing.assert_close(
        ProgramEmulator(edge.exported_program())(*inputs)[1],
        model.linear.bias,
        atol=0,
        rtol=0,
    )
    assert edge.to_executorch().buffer
