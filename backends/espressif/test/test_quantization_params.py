# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.passes.lower_to_esp import (
    _add_multipliers,
    _mul_multiplier,
    _QParams,
    LowerToEsp,
)
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
from executorch.backends.transforms.permute_pass_utils import set_arg
from executorch.backends.transforms.utils import add_constant
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind

_FUSED_ADD = exir_ops.edge.fused_quant.add.default
_ESP_ADD = exir_ops.edge.esp_nn.quantized_add_int8.default
_FUSED_MUL = exir_ops.edge.fused_quant.mul.default
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
            .export_fused_quant()
            .to_edge()
            .get_artifact()
            .exported_program()
        )
        for target, model in ((_FUSED_MUL, Multiply()), (_FUSED_ADD, Add()))
    }


@pytest.fixture
def multiply_program(exported_programs):
    return copy.deepcopy(exported_programs[_FUSED_MUL])


@pytest.fixture
def add_program(exported_programs):
    return copy.deepcopy(exported_programs[_FUSED_ADD])


@pytest.fixture(params=[_FUSED_MUL, _FUSED_ADD], ids=["mul", "add"])
def binary_program(exported_programs, request):
    return copy.deepcopy(exported_programs[request.param])


def _binary_node(program):
    return next(
        node for node in program.graph.nodes if node.target in (_FUSED_MUL, _FUSED_ADD)
    )


def _set_qparams(program, prefix, scale, zero_point, quant_min=-128, quant_max=127):
    node = _binary_node(program)
    for field, tensor in (
        ("scale", torch.tensor(scale, dtype=torch.float32)),
        ("zero_point", torch.tensor(zero_point, dtype=torch.int64)),
    ):
        constant = add_constant(
            program, f"test_{prefix}_{field}", tensor, node, InputKind.CONSTANT_TENSOR
        )
        set_arg(node, f"{prefix}_{field}", constant)
    set_arg(node, f"{prefix}_quant_min", quant_min)
    set_arg(node, f"{prefix}_quant_max", quant_max)
    program.graph_module.recompile()


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
    for prefix, scale, zero_point in (
        ("inp", 0.25, -17),
        ("other", 0.125, 23),
        ("out", 0.5, -7),
    ):
        _set_qparams(multiply_program, prefix, scale, zero_point)
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.ones(2, 3))
    expected = multiply_program.module()(*inputs)
    lowered = EspPassManager()(multiply_program).exported_program
    mul = next(node for node in lowered.graph.nodes if node.target == _ESP_MUL)
    assert mul.args[2:] == (17, -23, -7, 1 << 30, -3)
    assert all(arg.meta["val"].dtype == torch.int8 for arg in mul.args[:2])
    assert not any(node.target == _FUSED_MUL for node in lowered.graph.nodes)
    torch.testing.assert_close(lowered.module()(*inputs), expected, atol=0.5, rtol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_broadcast_keeps_qparams_with_operands(reverse):
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.linspace(-2, 2, 3))
    if reverse:
        inputs = inputs[::-1]
    program = (
        EspTester(Multiply(), inputs)
        .export_fused_quant()
        .to_edge()
        .get_artifact()
        .exported_program()
    )
    source_inputs = _binary_node(program).args[:2]
    for prefix, scale, zero_point in (
        ("inp", 0.01, -17),
        ("other", 0.02, 23),
        ("out", 0.5, -7),
    ):
        _set_qparams(program, prefix, scale, zero_point)
    expected = program.module()(*inputs)
    lowered = EspPassManager()(program).exported_program
    mul = next(node for node in lowered.graph.nodes if node.target == _ESP_MUL)
    assert [arg.name for arg in mul.args[:2]] == [
        arg.name for arg in (source_inputs[::-1] if reverse else source_inputs)
    ]
    assert mul.args[2:4] == ((-23, 17) if reverse else (17, -23))
    torch.testing.assert_close(lowered.module()(*inputs), expected, atol=0.5, rtol=0)


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
    for prefix in ("inp", "other"):
        _set_qparams(multiply_program, prefix, 1.0, input_zero_point)
    _set_qparams(multiply_program, "out", 1 / rescale, 127)
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
def test_unrepresentable_rescale_stays_portable(input_scale):
    assert (
        _mul_multiplier(
            _QParams(input_scale, 0), _QParams(input_scale, 0), _QParams(1.0, 0)
        )
        is None
    )


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
    for prefix, scale in zip(("inp", "other", "out"), scales):
        _set_qparams(multiply_program, prefix, scale, zero_point)
    assert LowerToEsp()(multiply_program).modified == supported


@pytest.mark.parametrize(
    "target, input_scale, output_scale, values, expected_values",
    [
        (_FUSED_MUL, 1e19, 1e38, [1.0, 2.0, 10.0], [1.0, 127.0, 127.0]),
        (_FUSED_MUL, 1e-20, 1e-40, [1.0, 2.0, 10.0], [127.0, 127.0, 127.0]),
        (_FUSED_ADD, 2e36, 4e36, [1.0, 10.0, 100.0], [1.0, 10.0, 127.0]),
    ],
    ids=["mul_product", "mul_reciprocal", "add"],
)
def test_float32_overflow_preserves_portable_results(
    exported_programs, target, input_scale, output_scale, values, expected_values
):
    program = copy.deepcopy(exported_programs[target])
    for prefix in ("inp", "other"):
        _set_qparams(program, prefix, input_scale, 0)
    _set_qparams(program, "out", output_scale, 0)
    for node in program.graph.nodes:
        if (
            node.target
            == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        ):
            node.args = (node.args[0], input_scale, 0, -128, 127, torch.int8)
        elif (
            node.target
            == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        ):
            node.args = (node.args[0], 1.0, 0, -128, 127, torch.int8)
    program.graph_module.recompile()
    inputs = torch.tensor([values, values]) * input_scale
    expected = torch.tensor([expected_values, expected_values])
    torch.testing.assert_close(
        program.module()(inputs, inputs), expected, atol=0, rtol=0
    )
    lowered = EspPassManager()(program).exported_program
    assert not any(node.target in (_ESP_MUL, _ESP_ADD) for node in lowered.graph.nodes)
    torch.testing.assert_close(
        lowered.module()(inputs, inputs), expected, atol=0, rtol=0
    )


@pytest.mark.parametrize("output_scale, supported", [(1e-38, True), (1e-40, False)])
def test_output_scale_reciprocal_limits(binary_program, output_scale, supported):
    input_scale = 1e-20 if _binary_node(binary_program).target == _FUSED_MUL else 1e-40
    for prefix in ("inp", "other"):
        _set_qparams(binary_program, prefix, input_scale, 0)
    _set_qparams(binary_program, "out", output_scale, 0)
    assert LowerToEsp()(binary_program).modified == supported


@pytest.mark.parametrize("prefix", ["inp", "other", "out"])
@pytest.mark.parametrize(
    "qparams",
    [
        (0.0, 0, -128, 127),
        (-1.0, 0, -128, 127),
        (float("nan"), 0, -128, 127),
        (float("inf"), 0, -128, 127),
        (1.0, -129, -128, 127),
        (1.0, 128, -128, 127),
        (1.0, 0, -129, 127),
        (1.0, 0, 1, -1),
    ],
)
def test_invalid_qparams_fail_export(binary_program, prefix, qparams):
    _set_qparams(binary_program, prefix, *qparams)
    with pytest.raises(
        ValueError, match=f"{_binary_node(binary_program).name}: {prefix}"
    ):
        LowerToEsp()(binary_program)


@pytest.mark.parametrize(
    "value",
    [
        torch.empty(3, 2, dtype=torch.int8, device="meta"),
        torch.empty(3, 2, dtype=torch.int8, device="meta").t(),
        torch.empty(0, dtype=torch.int8, device="meta"),
        torch.empty(1 << 31, dtype=torch.int8, device="meta"),
        torch.empty(2, 3, dtype=torch.uint8, device="meta"),
    ],
    ids=["shape_mismatch", "noncontiguous", "empty", "too_many_elements", "uint8"],
)
def test_unsupported_storage_stays_portable(binary_program, value):
    _binary_node(binary_program).args[0].meta["val"] = value
    assert not LowerToEsp()(binary_program).modified


@pytest.mark.parametrize(
    "prefix, dtype, quant_min, quant_max",
    [
        ("out", torch.uint8, 0, 255),
        ("out", torch.int8, -127, 127),
        ("inp", torch.float16, -128, 127),
    ],
)
def test_unsupported_quantization_stays_portable(
    binary_program, prefix, dtype, quant_min, quant_max
):
    _set_qparams(binary_program, prefix, 1.0, 0, quant_min, quant_max)
    set_arg(_binary_node(binary_program), f"{prefix}_dtype", dtype)
    assert not LowerToEsp()(binary_program).modified


@pytest.mark.parametrize("model", [Multiply, Add], ids=["mul", "add"])
def test_dynamic_shapes_stay_portable(model):
    inputs = (torch.ones(2, 3), torch.ones(2, 3))
    fused = EspTester(model(), inputs).export_fused_quant().get_artifact().module()
    batch = torch.export.Dim("batch", min=1, max=8)
    program = torch.export.export(
        fused, inputs, dynamic_shapes=({0: batch}, {0: batch})
    )
    edge = to_edge(program).exported_program()
    assert any(node.target in (_FUSED_MUL, _FUSED_ADD) for node in edge.graph.nodes)
    assert not LowerToEsp()(edge).modified


def test_lowering_is_idempotent(binary_program):
    first = EspPassManager()(binary_program)
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


def test_add_folding_and_replacement(add_program):
    for prefix, scale, zero_point in (
        ("inp", 0.25, -17),
        ("other", 0.125, 23),
        ("out", 0.5, -7),
    ):
        _set_qparams(add_program, prefix, scale, zero_point)
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.ones(2, 3))
    expected = add_program.module()(*inputs)
    lowered = EspPassManager()(add_program).exported_program
    add = next(node for node in lowered.graph.nodes if node.target == _ESP_ADD)
    assert add.args[2:] == (17, -23, -7, 1 << 30, 0, 1 << 30, -1, 20, 1 << 30, -19)
    assert all(arg.meta["val"].dtype == torch.int8 for arg in add.args[:2])
    assert not any(node.target == _FUSED_ADD for node in lowered.graph.nodes)
    torch.testing.assert_close(lowered.module()(*inputs), expected, atol=0.5, rtol=0)


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
    ],
)
def test_add_rescale_limits(scales, supported):
    assert (
        _add_multipliers(*(_QParams(scale, 0) for scale in scales)) is not None
    ) == supported


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
    for prefix, scale in zip(("inp", "other", "out"), scales):
        _set_qparams(add_program, prefix, scale, zero_point)
    assert LowerToEsp()(add_program).modified == supported


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
        input1, input2, *offsets, 1 << 30, 0, 1 << 30, -2, 20, 1 << 30, -18
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
