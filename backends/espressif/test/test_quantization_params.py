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
from executorch.backends.transforms.permute_pass_utils import set_arg
from executorch.backends.transforms.utils import add_constant
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind

_FUSED_MUL = exir_ops.edge.fused_quant.mul.default
_ESP_MUL = exir_ops.edge.esp_nn.quantized_mul_int8.default


class Multiply(torch.nn.Module):
    def forward(self, x, y):
        return x * y


@pytest.fixture(scope="module")
def exported_multiply():
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3), torch.ones(2, 3))
    return (
        EspTester(Multiply(), inputs)
        .export_fused_quant()
        .to_edge()
        .get_artifact()
        .exported_program()
    )


@pytest.fixture
def multiply_program(exported_multiply):
    return copy.deepcopy(exported_multiply)


def _mul_node(program):
    return next(node for node in program.graph.nodes if node.target == _FUSED_MUL)


def _set_qparams(program, prefix, scale, zero_point, quant_min=-128, quant_max=127):
    node = _mul_node(program)
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


def test_float32_overflow_preserves_portable_results(multiply_program):
    for prefix in ("inp", "other"):
        _set_qparams(multiply_program, prefix, 1e19, 0)
    _set_qparams(multiply_program, "out", 1e38, 0)
    for node in multiply_program.graph.nodes:
        if (
            node.target
            == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        ):
            node.args = (node.args[0], 1e19, 0, -128, 127, torch.int8)
        elif (
            node.target
            == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        ):
            node.args = (node.args[0], 1.0, 0, -128, 127, torch.int8)
    multiply_program.graph_module.recompile()
    inputs = torch.tensor([[1.0, 2.0, 10.0], [1.0, 2.0, 10.0]]) * 1e19
    expected = torch.tensor([[1.0, 127.0, 127.0], [1.0, 127.0, 127.0]])
    torch.testing.assert_close(
        multiply_program.module()(inputs, inputs), expected, atol=0, rtol=0
    )
    lowered = EspPassManager()(multiply_program).exported_program
    assert not any(node.target == _ESP_MUL for node in lowered.graph.nodes)
    torch.testing.assert_close(
        lowered.module()(inputs, inputs), expected, atol=0, rtol=0
    )


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
def test_invalid_qparams_fail_export(multiply_program, prefix, qparams):
    _set_qparams(multiply_program, prefix, *qparams)
    with pytest.raises(
        ValueError, match=f"{_mul_node(multiply_program).name}: {prefix}"
    ):
        LowerToEsp()(multiply_program)


@pytest.mark.parametrize(
    "value",
    [
        torch.empty(3, 2, dtype=torch.int8, device="meta"),
        torch.empty(3, 2, dtype=torch.int8, device="meta").t(),
        torch.empty(0, dtype=torch.int8, device="meta"),
        torch.empty(1 << 31, dtype=torch.int8, device="meta"),
        torch.empty(2, 3, dtype=torch.uint8, device="meta"),
    ],
    ids=["broadcast", "noncontiguous", "empty", "too_many_elements", "uint8"],
)
def test_unsupported_storage_stays_portable(multiply_program, value):
    _mul_node(multiply_program).args[0].meta["val"] = value
    assert not LowerToEsp()(multiply_program).modified


@pytest.mark.parametrize(
    "prefix, dtype, quant_min, quant_max",
    [
        ("out", torch.uint8, 0, 255),
        ("out", torch.int8, -127, 127),
        ("inp", torch.float16, -128, 127),
    ],
)
def test_unsupported_quantization_stays_portable(
    multiply_program, prefix, dtype, quant_min, quant_max
):
    _set_qparams(multiply_program, prefix, 1.0, 0, quant_min, quant_max)
    set_arg(_mul_node(multiply_program), f"{prefix}_dtype", dtype)
    assert not LowerToEsp()(multiply_program).modified


def test_dynamic_shapes_stay_portable():
    inputs = (torch.ones(2, 3), torch.ones(2, 3))
    fused = EspTester(Multiply(), inputs).export_fused_quant().get_artifact().module()
    batch = torch.export.Dim("batch", min=1, max=8)
    program = torch.export.export(
        fused, inputs, dynamic_shapes=({0: batch}, {0: batch})
    )
    edge = to_edge(program).exported_program()
    assert any(node.target == _FUSED_MUL for node in edge.graph.nodes)
    assert not LowerToEsp()(edge).modified


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
