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
from executorch.backends.test.suite.operators.test_add import Model as Add
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops

_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_MUL = exir_ops.edge.aten.mul.Tensor
_ADD = exir_ops.edge.aten.add.Tensor
_ESP_ADD = exir_ops.edge.esp_nn.quantized_add_int8.default
_ESP_MUL = exir_ops.edge.esp_nn.quantized_mul_int8.default


class Multiply(torch.nn.Module):
    def forward(self, x, y):
        return x * y


@pytest.fixture(scope="module")
def exported_programs():
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.ones(2, 3))
    return {
        target: (
            EspTester(model, inputs)
            .quantize()
            .export()
            .to_edge()
            .get_artifact()
            .exported_program()
        )
        for target, model in ((_MUL, Multiply()), (_ADD, Add()))
    }


@pytest.fixture
def multiply_program(exported_programs):
    return copy.deepcopy(exported_programs[_MUL])


@pytest.fixture
def add_program(exported_programs):
    return copy.deepcopy(exported_programs[_ADD])


@pytest.fixture(params=[_MUL, _ADD], ids=["mul", "add"])
def binary_program(exported_programs, request):
    return copy.deepcopy(exported_programs[request.param])


@pytest.fixture(scope="module")
def activation_programs():
    class BinaryActivation(torch.nn.Module):
        def __init__(self, operator, activation):
            super().__init__()
            self.operator = operator
            self.activation = activation

        def forward(self, x, y):
            return self.activation(self.operator(x, y))

    inputs = (torch.linspace(-4, 4, 17), torch.linspace(2, 8, 17))
    return {
        (operator, activation): (
            EspTester(BinaryActivation(operator, activation()), inputs)
            .quantize()
            .export()
            .to_edge()
            .get_artifact()
            .exported_program()
        )
        for operator in (torch.add, torch.mul)
        for activation in (torch.nn.ReLU, torch.nn.ReLU6)
    }


@pytest.fixture(
    params=[
        (torch.add, torch.nn.ReLU),
        (torch.add, torch.nn.ReLU6),
        (torch.mul, torch.nn.ReLU),
        (torch.mul, torch.nn.ReLU6),
    ],
    ids=["add_relu", "add_relu6", "mul_relu", "mul_relu6"],
)
def activation_program(activation_programs, request):
    return copy.deepcopy(activation_programs[request.param]), request.param[1]


def _pattern(graph_module):
    binary = next(
        node for node in graph_module.graph.nodes if node.target in (_MUL, _ADD)
    )
    output = binary
    if not any(user.target == _QUANTIZE for user in binary.users):
        output = next(
            user
            for user in binary.users
            if user.target
            in (exir_ops.edge.aten.relu.default, exir_ops.edge.aten.hardtanh.default)
        )
    quantize = next(node for node in output.users if node.target == _QUANTIZE)
    return *binary.args, quantize


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
    assert mul.args[2:] == (17, -23, -7, 1 << 30, -3, -128, 127)
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
        assert mul.args[-3] == expected_shift
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
    binary_program, operand, qparams, message
):
    node = _pattern(binary_program.graph_module)[operand]
    node.args = (node.args[0], *qparams, torch.int8)
    with pytest.raises(ValueError, match=f"{node.name}:.*{message}"):
        AtenToEspPass()(binary_program.graph_module)


@pytest.mark.parametrize(
    "value",
    [
        torch.empty(3, 2, dtype=torch.int8, device="meta"),
        torch.empty(3, 2, dtype=torch.int8, device="meta").t(),
        torch.empty(0, dtype=torch.int8, device="meta"),
        torch.empty(1 << 31, dtype=torch.int8, device="meta"),
    ],
    ids=["shape_mismatch", "noncontiguous", "empty", "too_many_elements"],
)
def test_unsupported_storage_stays_portable(binary_program, value):
    input1, _, output = _pattern(binary_program.graph_module)
    input1.args[0].meta["val"] = value
    assert not AtenToEspPass()(binary_program.graph_module).modified
    assert output in binary_program.graph.nodes


@pytest.mark.parametrize(
    "dtype, quant_min, quant_max", [(torch.uint8, 0, 255), (torch.int8, -127, 127)]
)
def test_unsupported_quantization_stays_portable(
    binary_program, dtype, quant_min, quant_max
):
    input1, _, output = _pattern(binary_program.graph_module)
    input1.args = (input1.args[0], 1.0, 0, quant_min, quant_max, dtype)
    assert not AtenToEspPass()(binary_program.graph_module).modified
    assert output in binary_program.graph.nodes


@pytest.mark.parametrize("model", [Multiply, Add], ids=["mul", "add"])
def test_dynamic_shapes_stay_portable(model):
    inputs = (torch.ones(2, 3), torch.ones(2, 3))
    quantized = EspTester(model(), inputs).quantize().get_artifact()
    batch = torch.export.Dim("batch", min=1, max=8)
    program = torch.export.export(
        quantized, inputs, dynamic_shapes=({0: batch}, {0: batch})
    )
    graph = to_edge(program).exported_program().graph_module
    _, _, output = _pattern(graph)
    assert not AtenToEspPass()(graph).modified
    assert output in graph.graph.nodes


def test_multiple_output_quantizers_stay_portable(binary_program):
    graph = binary_program.graph_module
    _, _, output = _pattern(graph)
    binary = output.args[0]
    with graph.graph.inserting_after(output):
        other = graph.graph.call_function(
            _QUANTIZE, (binary, 2.0, 13, -128, 127, torch.int8)
        )
        other.meta = output.meta.copy()
    graph_output = next(node for node in graph.graph.nodes if node.op == "output")
    graph_output.args = ((*graph_output.args[0], other),)
    graph.recompile()
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.ones(2, 3))
    expected = graph(*inputs)
    assert not AtenToEspPass()(graph).modified
    torch.testing.assert_close(graph(*inputs), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "operator, target", [(torch.mul, _ESP_MUL), (torch.add, _ESP_ADD)]
)
@pytest.mark.parametrize("float_user_first", [False, True])
@pytest.mark.parametrize(
    "activation",
    [None, torch.relu, torch.nn.functional.relu6],
    ids=["none", "relu", "relu6"],
)
def test_direct_float_branch_is_preserved(
    float_user_first, operator, target, activation
):
    class BranchedBinary(torch.nn.Module):
        def forward(self, x, y):
            result = operator(x, y)
            if activation is not None:
                result = activation(result)
            return result, torch.sin(result)

    inputs = (
        torch.linspace(-1, 1, 6).reshape(2, 3),
        torch.linspace(-2, 2, 6).reshape(2, 3),
    )
    tester = EspTester(BranchedBinary(), inputs).quantize().export().to_edge()
    graph = tester.get_artifact().exported_program().graph_module
    input1, input2, quantize = _pattern(graph)
    binary = next(node for node in graph.graph.nodes if node.target in (_MUL, _ADD))
    float_user = next(
        node
        for node in graph.graph.nodes
        if node.target == exir_ops.edge.aten.sin.default
    )
    # PT2E normally puts sin after output DQ; exercise a direct float consumer.
    float_user.args = (binary,)
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
    esp_binary = next(node for node in lowered.graph.nodes if node.target == target)
    assert quantize not in lowered.graph.nodes
    assert binary in lowered.graph.nodes
    assert binary.args == (input1, input2)
    assert float_user.args == (binary,)
    output_dequantize = next(iter(esp_binary.users))
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


def test_lowering_is_idempotent(binary_program):
    first = EspPassManager()(binary_program)
    assert first.modified
    assert not EspPassManager()(first.exported_program).modified


@pytest.mark.parametrize(
    "scale, zero_point, relu6_max",
    [(0.5, -7, 5), (4.0, -7, -5), (12.0, -7, -7), (0.01, -128, 127)],
    ids=["bounded", "round_up", "round_down", "saturated"],
)
def test_activation_bounds_use_output_quantization(
    activation_program, scale, zero_point, relu6_max
):
    program, activation = activation_program
    graph = program.graph_module
    *_, quantize = _pattern(graph)
    quantize.args = (quantize.args[0], scale, zero_point, -128, 127, torch.int8)
    assert AtenToEspPass()(graph).modified
    binary = next(
        node for node in graph.graph.nodes if node.target in (_ESP_ADD, _ESP_MUL)
    )
    maximum = relu6_max if activation is torch.nn.ReLU6 else 127
    assert binary.args[-2:] == (zero_point, maximum)
    assert quantize not in graph.graph.nodes


def test_activation_clamps_ignore_default_dtype(activation_program):
    program, _ = activation_program
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert AtenToEspPass()(program.graph_module).modified
    finally:
        torch.set_default_dtype(default_dtype)


def test_multiple_activation_quantizers_stay_portable(activation_program):
    program, _ = activation_program
    graph = program.graph_module
    *_, quantize = _pattern(graph)
    with graph.graph.inserting_after(quantize):
        other = graph.graph.call_function(
            _QUANTIZE, (quantize.args[0], 2.0, 13, -128, 127, torch.int8)
        )
        other.meta = quantize.meta.copy()
    output = next(node for node in graph.graph.nodes if node.op == "output")
    output.args = ((*output.args[0], other),)
    graph.recompile()
    assert not AtenToEspPass()(graph).modified
    assert quantize in graph.graph.nodes and other in graph.graph.nodes


@pytest.mark.parametrize("operator", [torch.add, torch.mul], ids=["add", "mul"])
def test_other_hardtanh_bounds_stay_portable(activation_programs, operator):
    program = copy.deepcopy(activation_programs[(operator, torch.nn.ReLU6)])
    graph = program.graph_module
    *_, quantize = _pattern(graph)
    activation = quantize.args[0]
    activation.args = (activation.args[0], -1.0, 1.0)
    assert not AtenToEspPass()(graph).modified
    assert quantize in graph.graph.nodes and activation in graph.graph.nodes


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


def test_add_folding_and_replacement(add_program):
    graph = add_program.graph_module
    for node, scale, zero_point in zip(
        _pattern(graph), (0.25, 0.125, 0.5), (-17, 23, -7)
    ):
        node.args = (node.args[0], scale, zero_point, -128, 127, torch.int8)
    assert AtenToEspPass()(graph).modified
    add = next(node for node in graph.graph.nodes if node.target == _ESP_ADD)
    assert add.args[2:] == (
        17,
        -23,
        -7,
        1 << 30,
        0,
        1 << 30,
        -1,
        20,
        1 << 30,
        -19,
        -128,
        127,
    )


@pytest.mark.parametrize(
    "scales, supported",
    [
        ((1, 1, 2**-19), False),
        ((1, 1, 2**-18), True),
        ((1, 1, 2**12), True),
        ((1, 1, 2**13), False),
        ((2**-30, 1, 1), True),
        ((2**-31, 1, 1), False),
        ((1, 2**-31, 1), False),
        ((1e-200, 1e-200, 1), False),
        ((1e200, 1e200, 1), False),
        ((1e-40, 1e-40, 1e-40), False),
        ((1e-200, 1e-200, 1e-200), False),
    ],
)
def test_add_rescale_limits(add_program, scales, supported):
    graph = add_program.graph_module
    for node, scale in zip(_pattern(graph), scales):
        node.args = (node.args[0], scale, 0, -128, 127, torch.int8)
    assert AtenToEspPass()(graph).modified == supported


@pytest.mark.parametrize(
    "scales, zero_point, supported",
    [
        ((1e38, 1e38, 1e38), 0, False),
        ((2e36, 2e36, 4e36), 0, False),
        ((1e36, 1e36, 2e36), 0, True),
        ((1e36, 1e36, 2e36), -128, False),
    ],
    ids=["dequantization", "sum", "finite", "offset"],
)
def test_add_float32_range_limits(add_program, scales, zero_point, supported):
    graph = add_program.graph_module
    for node, scale in zip(_pattern(graph), scales):
        node.args = (node.args[0], scale, zero_point, -128, 127, torch.int8)
    assert AtenToEspPass()(graph).modified == supported


def test_add_float32_overflow_preserves_portable_results(add_program):
    graph = add_program.graph_module
    input1, input2, output = _pattern(graph)
    for dequantize in (input1, input2):
        for node in (dequantize.args[0], dequantize):
            node.args = (node.args[0], 2e36, 0, -128, 127, torch.int8)
    output.args = (output.args[0], 4e36, 0, -128, 127, torch.int8)
    graph_output = next(node for node in graph.graph.nodes if node.op == "output")
    graph_output.args = ((output,),)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    inputs = torch.tensor([[1.0, 10.0, 100.0], [1.0, 10.0, 100.0]]) * 2e36
    expected = torch.tensor([[1, 10, 127], [1, 10, 127]], dtype=torch.int8)
    torch.testing.assert_close(graph(inputs, inputs)[0], expected, atol=0, rtol=0)
    assert not AtenToEspPass()(graph).modified
    assert output in graph.graph.nodes
    torch.testing.assert_close(graph(inputs, inputs)[0], expected, atol=0, rtol=0)


@pytest.mark.parametrize("offsets", [(0, 0, 0), (128, -127, -128), (-127, 128, 127)])
def test_add_matches_existing_integer_reference(offsets):
    values = torch.arange(-128, 128, dtype=torch.int8)
    input1, input2 = torch.meshgrid(values, values, indexing="ij")
    input1_offset, input2_offset, output_offset = offsets
    lhs = (input1.numpy().astype("int64") + input1_offset) << 20
    rhs = (input2.numpy().astype("int64") + input2_offset) << 20
    total = multiply_by_quantized_multiplier(lhs, 1 << 30, 0) + (
        multiply_by_quantized_multiplier(rhs, 1 << 30, -2)
    )
    expected = multiply_by_quantized_multiplier(total, 1 << 30, -18) + output_offset
    expected = torch.from_numpy(expected.clip(-128, 127)).to(torch.int8)
    actual = torch.ops.esp_nn.quantized_add_int8(
        input1, input2, *offsets, 1 << 30, 0, 1 << 30, -2, 20, 1 << 30, -18, -128, 127
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.fixture(scope="module")
def linear_program():
    torch.manual_seed(17)
    return (
        EspTester(torch.nn.Linear(17, 5), (torch.linspace(-3, 5, 34).reshape(2, 17),))
        .quantize()
        .export()
        .to_edge()
        .get_artifact()
        .exported_program()
    )


def test_linear_parameters_are_precomputed(linear_program):
    original = copy.deepcopy(linear_program)
    graph = original.graph
    linear = next(
        n for n in graph.nodes if n.target == exir_ops.edge.aten.addmm.default
    )
    bias, input_dq, transpose = linear.args
    weight_dq = transpose.args[0]
    scales = get_param_tensor(original, weight_dq.args[1]).to(torch.float64)
    output_q = next(n for n in linear.users if n.target == _QUANTIZE)
    expected_bias = torch.round(
        get_param_tensor(original, bias).to(torch.float64) / (input_dq.args[1] * scales)
    ).to(torch.int32)
    rescales = [
        quantize_multiplier(input_dq.args[1] * scale / output_q.args[1])
        for scale in scales.tolist()
    ]
    result = EspPassManager()(original).exported_program
    lowered = next(
        n
        for n in result.graph.nodes
        if n.target == exir_ops.edge.esp_nn.quantized_linear_int8.default
    )
    torch.testing.assert_close(get_param_tensor(result, lowered.args[2]), expected_bias)
    for index, values in enumerate(zip(*rescales), 3):
        torch.testing.assert_close(
            get_param_tensor(result, lowered.args[index]),
            torch.tensor(values, dtype=torch.int32),
        )
    assert all(value.dtype != torch.float32 for value in result.state_dict.values())
    assert all(value.dtype == torch.int32 for value in result.constants.values())
    assert not EspPassManager()(result).modified


@pytest.mark.parametrize(
    "unsupported",
    [
        "alpha",
        "beta",
        "weight_zero_point",
        "bias_overflow",
        "positive_shift_overflow",
        "negative_shift",
        "float_input_overflow",
        "float_weight_overflow",
        "float_accumulator_overflow",
    ],
)
def test_linear_unsupported_parameters_stay_portable(linear_program, unsupported):
    program = copy.deepcopy(linear_program)
    linear = next(
        n for n in program.graph.nodes if n.target == exir_ops.edge.aten.addmm.default
    )
    bias, input_dq, transpose = linear.args
    weight_dq = transpose.args[0]
    if unsupported in ("alpha", "beta"):
        linear.kwargs = {unsupported: 2}
    elif unsupported == "weight_zero_point":
        get_param_tensor(program, weight_dq.args[2]).fill_(1)
    elif unsupported == "bias_overflow":
        get_param_tensor(program, bias).data.fill_(1e20)
    elif unsupported.startswith("float_"):
        input_scale, weight_scale, output_scale = {
            "float_input_overflow": (1e38, 1e-38, 1.0),
            "float_weight_overflow": (1e-38, 1e38, 1.0),
            "float_accumulator_overflow": (1e17, 1e17, 1e34),
        }[unsupported]
        input_dq.args = (input_dq.args[0], input_scale, 0, -128, 127, torch.int8)
        get_param_tensor(program, weight_dq.args[1]).fill_(weight_scale)
        quantize = next(n for n in linear.users if n.target == _QUANTIZE)
        quantize.args = (quantize.args[0], output_scale, 0, -128, 127, torch.int8)
    else:
        get_param_tensor(program, weight_dq.args[1]).fill_(1.0)
        get_param_tensor(program, bias).data.zero_()
        input_dq.args = (input_dq.args[0], 1.0, 0, -128, 127, torch.int8)
        quantize = next(n for n in linear.users if n.target == _QUANTIZE)
        scale = 2.0**32 if unsupported == "negative_shift" else 2.0**-20
        quantize.args = (quantize.args[0], scale, 0, -128, 127, torch.int8)
    program.graph_module.recompile()
    result = EspPassManager()(program)
    assert not result.modified
    assert not any(
        n.target == exir_ops.edge.esp_nn.quantized_linear_int8.default
        for n in result.exported_program.graph.nodes
    )


def test_linear_preserves_direct_float_users():
    class BranchedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(17, 5)

        def forward(self, x):
            result = self.linear(x)
            return result, torch.sin(result)

    torch.manual_seed(17)
    inputs = (torch.linspace(-3, 5, 34).reshape(2, 17),)
    tester = EspTester(BranchedLinear(), inputs).quantize().export().to_edge()
    program = tester.get_artifact().exported_program()
    linear = next(
        n for n in program.graph.nodes if n.target == exir_ops.edge.aten.addmm.default
    )
    sine = next(
        n for n in program.graph.nodes if n.target == exir_ops.edge.aten.sin.default
    )
    sine.args = (linear,)
    program.graph_module.recompile()
    expected = program.module()(*inputs)
    tester.run_passes()
    result = tester.get_artifact().exported_program()
    actual = result.module()(*inputs)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    assert any(n.target == exir_ops.edge.aten.addmm.default for n in result.graph.nodes)
    assert any(
        n.target == exir_ops.edge.esp_nn.quantized_linear_int8.default
        for n in result.graph.nodes
    )
    assert not EspPassManager()(result).modified
    tester.to_executorch().serialize()


@pytest.mark.parametrize("features", [(65536, 1), (1, 65536)])
def test_linear_vendor_dimension_limits(features):
    in_features, out_features = features
    program = (
        EspTester(
            torch.nn.Linear(in_features, out_features), (torch.ones(1, in_features),)
        )
        .quantize()
        .export()
        .to_edge()
        .get_artifact()
        .exported_program()
    )
    assert not EspPassManager()(program).modified
