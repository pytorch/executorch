# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.passes.aten_to_esp_pass import AtenToEspPass
from executorch.backends.espressif.reference.esp_nn_reference import (
    multiply_by_quantized_multiplier,
)
from executorch.backends.espressif.serialization.xts3 import _quantize_multiplier
from executorch.backends.espressif.test.tester import EspTester
from executorch.backends.espressif.utils.quantization import (
    quantize_multiplier,
    requantize,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops

_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_MUL = exir_ops.edge.aten.mul.Tensor
_ESP_MUL = exir_ops.edge.esp_nn.quantized_mul_int8.default


class Multiply(torch.nn.Module):
    def forward(self, x, y):
        return x * y


@pytest.fixture(scope="module")
def exported_multiply():
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.ones(2, 3))
    return (
        EspTester(Multiply(), inputs)
        .quantize()
        .export()
        .to_edge()
        .get_artifact()
        .exported_program()
    )


@pytest.fixture
def multiply_program(exported_multiply):
    return copy.deepcopy(exported_multiply)


def _pattern(graph_module):
    mul = next(node for node in graph_module.graph.nodes if node.target == _MUL)
    quantize = next(node for node in mul.users if node.target == _QUANTIZE)
    return *mul.args, quantize


@pytest.mark.parametrize(
    "scale, expected",
    [
        (2**-32, (1 << 30, -31)),
        (2**-31, (1 << 30, -30)),
        (0.25, (1 << 30, -1)),
        (0.5, (1 << 30, 0)),
        (1.0, (1 << 30, 1)),
        (2**29, (1 << 30, 30)),
        ((2**30 + 0.5) / 2**31, (1 << 30, 0)),
        ((2**30 + 1.5) / 2**31, ((1 << 30) + 2, 0)),
        (1 - 2**-32, (1 << 30, 1)),
    ],
)
def test_multiplier_encoding_preserves_delegate_rounding(scale, expected):
    assert quantize_multiplier(scale) == expected
    assert _quantize_multiplier(scale) == expected


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_multiplier_requires_finite_positive_scale(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        quantize_multiplier(scale)


@pytest.mark.parametrize("scale", [2**-33, 2**30])
def test_multiplier_shift_range(scale):
    with pytest.raises(ValueError, match="shift range"):
        quantize_multiplier(scale)


def test_folding_and_replacement(multiply_program):
    graph = multiply_program.graph_module
    input1, input2, output = _pattern(graph)
    for node, scale, zero_point in (
        (input1, 0.25, -17),
        (input2, 0.125, 23),
        (output, 0.5, -7),
    ):
        node.args = (node.args[0], scale, zero_point, -128, 127, torch.int8)
    graph.recompile()
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.ones(2, 3))
    expected = graph(*inputs)
    lowered = AtenToEspPass()(graph).graph_module
    mul = next(node for node in lowered.graph.nodes if node.target == _ESP_MUL)
    assert mul.args[2:] == (17, -23, -7, 1 << 30, -3)
    assert all(arg.meta["val"].dtype == torch.int8 for arg in mul.args[:2])
    assert output not in lowered.graph.nodes
    assert not any(node.target == _MUL for node in lowered.graph.nodes)
    torch.testing.assert_close(lowered(*inputs), expected, atol=0.5, rtol=0)


@pytest.mark.parametrize(
    "rescale, input_zero_point, expected_shift",
    [
        (2**-32, 0, None),
        (2**-31, 0, -30),
        (2**15, 0, 16),
        (2**16, 0, None),
        (2**14, -128, 15),
        (2**15, -128, None),
        (2**29, 0, None),
        (2**30, 0, None),
    ],
)
def test_kernel_shift_and_overflow_limits(
    multiply_program, rescale, input_zero_point, expected_shift
):
    input1, input2, output = _pattern(multiply_program.graph_module)
    for node in (input1, input2):
        node.args = (node.args[0], 1.0, input_zero_point, -128, 127, torch.int8)
    output.args = (output.args[0], 1 / rescale, 127, -128, 127, torch.int8)
    result = EspPassManager()(multiply_program)
    mul = next(
        (
            node
            for node in result.exported_program.graph.nodes
            if node.target == _ESP_MUL
        ),
        None,
    )
    if expected_shift is None:
        assert mul is None
    else:
        assert mul.args[-1] == expected_shift
        assert mul.args[4] == 127


@pytest.mark.parametrize("input_scale", [1e-200, 1e200])
def test_unrepresentable_rescale_stays_portable(multiply_program, input_scale):
    input1, input2, output = _pattern(multiply_program.graph_module)
    for node in (input1, input2):
        node.args = (node.args[0], input_scale, 0, -128, 127, torch.int8)
    output.args = (output.args[0], 1.0, 0, -128, 127, torch.int8)
    assert not AtenToEspPass()(multiply_program.graph_module).modified
    assert output in multiply_program.graph.nodes


@pytest.mark.parametrize(
    "scales, zero_point, supported",
    [
        ((1e38, 1e-38, 1.0), 0, False),
        ((1e-38, 1e38, 1.0), 0, False),
        ((1e19, 1e19, 1e38), 0, False),
        ((1e17, 1e17, 1e34), 0, True),
        ((1e17, 1e17, 1e34), -128, False),
        (
            (1.2e17, torch.finfo(torch.float32).max / (128 * 128 * 1.2e17), 1e34),
            0,
            False,
        ),
    ],
    ids=["input1", "input2", "product", "finite", "offset", "float32_rounding"],
)
def test_float32_range_limits(multiply_program, scales, zero_point, supported):
    graph = multiply_program.graph_module
    for node, scale in zip(_pattern(graph), scales):
        node.args = (node.args[0], scale, zero_point, -128, 127, torch.int8)
    assert AtenToEspPass()(graph).modified == supported


def test_float32_overflow_preserves_portable_results(multiply_program):
    graph = multiply_program.graph_module
    input1, input2, output = _pattern(graph)
    for dequantize in (input1, input2):
        quantize = dequantize.args[0]
        for node in (quantize, dequantize):
            node.args = (node.args[0], 1e19, 0, -128, 127, torch.int8)
    output.args = (output.args[0], 1e38, 0, -128, 127, torch.int8)
    graph_output = next(node for node in graph.graph.nodes if node.op == "output")
    graph_output.args = ((output,),)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    inputs = torch.tensor([[1.0, 2.0, 10.0], [1.0, 2.0, 10.0]]) * 1e19
    expected = torch.tensor([[1, 127, 127], [1, 127, 127]], dtype=torch.int8)
    torch.testing.assert_close(graph(inputs, inputs)[0], expected, atol=0, rtol=0)

    assert not AtenToEspPass()(graph).modified
    assert output in graph.graph.nodes
    torch.testing.assert_close(graph(inputs, inputs)[0], expected, atol=0, rtol=0)


@pytest.mark.parametrize("operand", [0, 1, 2])
@pytest.mark.parametrize(
    "qparams, message",
    [
        ((0.0, 0, -128, 127), "finite and positive"),
        ((-1.0, 0, -128, 127), "finite and positive"),
        ((float("nan"), 0, -128, 127), "finite and positive"),
        ((float("inf"), 0, -128, 127), "finite and positive"),
        ((1.0, -129, -128, 127), "range or zero point"),
        ((1.0, 128, -128, 127), "range or zero point"),
        ((1.0, 0, -129, 127), "range or zero point"),
        ((1.0, 0, 1, -1), "range or zero point"),
    ],
)
def test_malformed_qparams_fail_with_node_name(
    multiply_program, operand, qparams, message
):
    node = _pattern(multiply_program.graph_module)[operand]
    node.args = (node.args[0], *qparams, torch.int8)
    with pytest.raises(ValueError, match=f"{node.name}:.*{message}"):
        AtenToEspPass()(multiply_program.graph_module)


@pytest.mark.parametrize(
    "value",
    [
        torch.empty(3, 2, dtype=torch.int8, device="meta"),
        torch.empty(3, 2, dtype=torch.int8, device="meta").t(),
        torch.empty(0, dtype=torch.int8, device="meta"),
        torch.empty(1 << 31, dtype=torch.int8, device="meta"),
    ],
    ids=["broadcast", "noncontiguous", "empty", "too_many_elements"],
)
def test_unsupported_storage_stays_portable(multiply_program, value):
    input1, _, output = _pattern(multiply_program.graph_module)
    input1.args[0].meta["val"] = value
    assert not AtenToEspPass()(multiply_program.graph_module).modified
    assert output in multiply_program.graph.nodes


@pytest.mark.parametrize(
    "dtype, quant_min, quant_max", [(torch.uint8, 0, 255), (torch.int8, -127, 127)]
)
def test_unsupported_quantization_stays_portable(
    multiply_program, dtype, quant_min, quant_max
):
    input1, _, output = _pattern(multiply_program.graph_module)
    input1.args = (input1.args[0], 1.0, 0, quant_min, quant_max, dtype)
    assert not AtenToEspPass()(multiply_program.graph_module).modified
    assert output in multiply_program.graph.nodes


def test_dynamic_shapes_stay_portable():
    inputs = (torch.ones(2, 3), torch.ones(2, 3))
    quantized = EspTester(Multiply(), inputs).quantize().get_artifact()
    batch = torch.export.Dim("batch", min=1, max=8)
    program = torch.export.export(
        quantized, inputs, dynamic_shapes=({0: batch}, {0: batch})
    )
    graph = to_edge(program).exported_program().graph_module
    _, _, output = _pattern(graph)
    assert not AtenToEspPass()(graph).modified
    assert output in graph.graph.nodes


def test_multiple_output_quantizers_stay_portable(multiply_program):
    graph = multiply_program.graph_module
    _, _, output = _pattern(graph)
    mul = output.args[0]
    with graph.graph.inserting_after(output):
        other = graph.graph.call_function(
            _QUANTIZE, (mul, 2.0, 13, -128, 127, torch.int8)
        )
        other.meta = output.meta.copy()
    graph_output = next(node for node in graph.graph.nodes if node.op == "output")
    graph_output.args = ((*graph_output.args[0], other),)
    graph.recompile()
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.ones(2, 3))
    expected = graph(*inputs)
    assert not AtenToEspPass()(graph).modified
    torch.testing.assert_close(graph(*inputs), expected, atol=0, rtol=0)


@pytest.mark.parametrize("float_user_first", [False, True])
def test_direct_float_branch_is_preserved(float_user_first):
    class BranchedMultiply(torch.nn.Module):
        def forward(self, x, y):
            product = x * y
            return product, torch.sin(product)

    inputs = (
        torch.linspace(-1, 1, 6).reshape(2, 3),
        torch.linspace(-2, 2, 6).reshape(2, 3),
    )
    tester = EspTester(BranchedMultiply(), inputs).quantize().export().to_edge()
    graph = tester.get_artifact().exported_program().graph_module
    input1, input2, quantize = _pattern(graph)
    mul = quantize.args[0]
    float_user = next(
        node
        for node in graph.graph.nodes
        if node.target == exir_ops.edge.aten.sin.default
    )
    # PT2E normally puts sin after output DQ; exercise a direct float consumer.
    float_user.args = (mul,)
    if float_user_first:
        with graph.graph.inserting_after(quantize):
            new_quantize = graph.graph.node_copy(quantize)
        quantize.replace_all_uses_with(new_quantize)
        graph.graph.erase_node(quantize)
        quantize = new_quantize
    graph.recompile()
    expected = graph(*inputs)
    assert not torch.equal(torch.sin(expected[0]), expected[1])

    tester.run_passes()
    program = tester.get_artifact().exported_program()
    lowered = program.graph_module
    esp_mul = next(node for node in lowered.graph.nodes if node.target == _ESP_MUL)
    assert quantize not in lowered.graph.nodes
    assert mul in lowered.graph.nodes
    assert mul.args == (input1, input2)
    assert float_user.args == (mul,)
    output_dequantize = next(iter(esp_mul.users))
    assert output_dequantize.target == (
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
    )
    actual = lowered(*inputs)
    torch.testing.assert_close(
        actual[0], expected[0], atol=output_dequantize.args[1], rtol=0
    )
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    assert not EspPassManager()(program).modified
    tester.to_executorch().serialize()


def test_lowering_is_idempotent(multiply_program):
    first = EspPassManager()(multiply_program)
    assert first.modified
    assert not EspPassManager()(first.exported_program).modified


@pytest.mark.parametrize("scale", [0.03, 0.25, 0.5, 0.7, 1.0, 1.3, 2.0, 32768.0])
def test_requantization_matches_existing_integer_reference(scale):
    values = torch.arange(-128, 128, dtype=torch.int64)
    products = (values[:, None] * values[None, :]).flatten()
    multiplier, shift = quantize_multiplier(scale)
    expected = torch.from_numpy(
        multiply_by_quantized_multiplier(products.numpy(), multiplier, shift)
    ).to(torch.int32)
    torch.testing.assert_close(
        requantize(products, multiplier, shift), expected, atol=0, rtol=0
    )
