# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.ao_ext.mxfp import (
    _cast_to_block_scaled_cpu_ref,
    _quantize_tosa_reference,
)
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    to_dtype,
    to_mx,
)


# Finite maxima and upper rounding boundaries are independent format constants.
_FORMATS = [
    pytest.param(torch.float4_e2m1fn_x2, 6.0, 7.0, False, id="fp4e2m1"),
    pytest.param(DTYPE_FP6_E2M3, 7.5, 7.75, False, id="fp6e2m3"),
    pytest.param(DTYPE_FP6_E3M2, 28.0, 30.0, False, id="fp6e3m2"),
    pytest.param(torch.float8_e4m3fn, 448.0, 464.0, True, id="fp8e4m3"),
    pytest.param(torch.float8_e5m2, 57344.0, 61440.0, False, id="fp8e5m2"),
]


@pytest.mark.parametrize("output_dtype,maximum,limit,inclusive", _FORMATS)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("exponent", [-120, 0, 100])
def test_scale_and_payload_transition(
    output_dtype, maximum, limit, inclusive, input_dtype, exponent
):
    boundary = torch.tensor(limit * 2.0**exponent, dtype=input_dtype)
    maxima = torch.stack(
        [
            torch.nextafter(boundary, torch.tensor(0.0, dtype=input_dtype)),
            boundary,
            torch.nextafter(boundary, torch.tensor(float("inf"), dtype=input_dtype)),
        ]
    )
    for sign in (1, -1):
        blocks = (maxima * sign)[:, None].expand(-1, 32)
        payload, scales = _quantize_tosa_reference(blocks, output_dtype, 32)
        assert scales.view(torch.uint8).flatten().tolist() == [
            127 + exponent,
            127 + exponent if inclusive else 128 + exponent,
            128 + exponent,
        ]

        if sign == 1:
            expected = {
                torch.float4_e2m1fn_x2: [0x77, 0x66, 0x66],
                DTYPE_FP6_E2M3: [31, 24, 24],
                DTYPE_FP6_E3M2: [31, 28, 28],
                torch.float8_e4m3fn: [126, 126, 119],
                torch.float8_e5m2: [123, 120, 120],
            }[output_dtype]
            assert payload.view(torch.uint8)[:, 0].tolist() == expected


@pytest.mark.parametrize(
    "output_dtype,maximum,expected,rounding",
    [
        (
            torch.float4_e2m1fn_x2,
            6.0,
            [0x80, 0x91, 0xA2, 0x32, 0xC4, 0xF7],
            (1.25, 1.75, 0.5),
        ),
        (
            DTYPE_FP6_E2M3,
            7.5,
            [0, 32, 4, 36, 8, 40, 10, 12, 18, 50, 31, 63],
            (1.0625, 1.1875, 0.125),
        ),
        (
            DTYPE_FP6_E3M2,
            28.0,
            [0, 32, 8, 40, 12, 44, 13, 14, 17, 49, 31, 63],
            (1.125, 1.375, 0.0625),
        ),
        (
            torch.float8_e4m3fn,
            448.0,
            [0, 128, 48, 176, 56, 184, 58, 60, 66, 194, 126, 254],
            (1.0625, 1.1875, 2.0**-9),
        ),
        (
            torch.float8_e5m2,
            57344.0,
            [0, 128, 56, 184, 60, 188, 61, 62, 65, 193, 123, 251],
            (1.125, 1.375, 2.0**-16),
        ),
    ],
)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_payload_bytes_and_rounding(
    output_dtype, maximum, expected, rounding, input_dtype
):
    tie, second_tie, subnormal = rounding
    boundary = torch.tensor(tie, dtype=input_dtype)
    values = torch.stack(
        [
            torch.nextafter(boundary, torch.tensor(0.0, dtype=input_dtype)),
            boundary,
            torch.nextafter(boundary, torch.tensor(float("inf"), dtype=input_dtype)),
            torch.tensor(second_tie, dtype=input_dtype),
            torch.tensor(subnormal / 2, dtype=input_dtype),
            torch.tensor(subnormal * 1.5, dtype=input_dtype),
        ]
    )
    data = torch.zeros(32, dtype=input_dtype)
    data[:12] = torch.tensor(
        [0.0, -0.0, 0.5, -0.5, 1, -1, 1.25, 1.5, 2.5, -2.5, maximum, -maximum]
    )
    data[12:18] = values
    data[18:24] = -values
    payload, scales = _quantize_tosa_reference(data, output_dtype, 32)
    if output_dtype == torch.float4_e2m1fn_x2:
        expected = [code for byte in expected for code in (byte & 0xF, byte >> 4)]
    unit_code, sign_bit = expected[4], expected[1]
    rounded = [unit_code, unit_code, unit_code + 1, unit_code + 2, 0, 2]
    expected = expected + rounded + [code | sign_bit for code in rounded] + [0] * 8
    if output_dtype == torch.float4_e2m1fn_x2:
        expected = [
            low | (high << 4) for low, high in zip(expected[::2], expected[1::2])
        ]
    assert scales.view(torch.uint8).tolist() == [127]
    assert payload.view(torch.uint8).tolist() == expected


@pytest.mark.parametrize("output_dtype,maximum,limit,inclusive", _FORMATS)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("channels_last", [False, True])
def test_independent_blocks_and_noncontiguous_input(
    output_dtype, maximum, limit, inclusive, input_dtype, channels_last
):
    data = (
        torch.zeros((64, 2, 2), dtype=input_dtype).permute(1, 2, 0)
        if channels_last
        else torch.zeros((2, 2, 128), dtype=input_dtype)[..., ::2]
    )
    blocks = data.view(2, 2, 2, 32)
    blocks[..., 0] = maximum * torch.tensor([0.5, 1, 2, 4, 8, 16, 32, 64]).view(2, 2, 2)
    assert not data.is_contiguous()
    payload, scales = _quantize_tosa_reference(data, output_dtype, 32)
    assert scales.shape == (2, 2, 2)
    assert scales.view(torch.uint8).flatten().tolist() == list(range(126, 134))
    assert payload.shape == (2, 2, 32 if output_dtype == torch.float4_e2m1fn_x2 else 64)
    assert payload.is_contiguous()
    actual = to_dtype(payload, scales, output_dtype, 32, torch.float32)
    torch.testing.assert_close(actual, data.float(), rtol=0, atol=0)


@pytest.mark.parametrize("output_dtype,maximum,limit,inclusive", _FORMATS)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_zero_and_nonfinite_blocks(
    output_dtype, maximum, limit, inclusive, input_dtype
):
    data = torch.zeros((5, 32), dtype=input_dtype)
    data[0] = -0.0
    data[2, 0] = float("inf")
    data[3, 0] = -float("inf")
    data[4, 0] = float("nan")
    payload, scales = _quantize_tosa_reference(data, output_dtype, 32)
    assert scales.view(torch.uint8).flatten().tolist() == [0, 0, 255, 255, 255]
    actual = to_dtype(payload, scales, output_dtype, 32, torch.float32)
    assert torch.isnan(actual[2:]).all()
    assert (actual[:2] == 0).all()
    assert torch.signbit(actual[0]).all()
    assert not torch.signbit(actual[1]).any()


@pytest.mark.parametrize(
    "output_dtype,largest_scale",
    [
        (torch.float4_e2m1fn_x2, 253),
        (DTYPE_FP6_E2M3, 253),
        (DTYPE_FP6_E3M2, 251),
        (torch.float8_e4m3fn, 247),
        (torch.float8_e5m2, 240),
    ],
)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_extreme_finite_blocks(output_dtype, largest_scale, input_dtype):
    tiny = torch.tensor(torch.finfo(input_dtype).tiny, dtype=input_dtype)
    smallest = torch.nextafter(torch.tensor(0.0, dtype=input_dtype), tiny)
    maxima = torch.stack([smallest, tiny, torch.tensor(torch.finfo(input_dtype).max)])
    data = maxima[:, None].expand(-1, 32).to(input_dtype)
    payload, scales = _quantize_tosa_reference(data, output_dtype, 32)
    assert scales.view(torch.uint8).flatten().tolist() == [0, 0, largest_scale]
    actual = to_dtype(payload, scales, output_dtype, 32, torch.float32)
    assert torch.isfinite(actual[:2]).all()
    # The largest finite input rounds up to 2**128, overflowing FP32.
    assert torch.isposinf(actual[2]).all()


def test_fp4_reference_policy_differs_from_rceil():
    data = torch.zeros(32)
    data[:2] = torch.tensor([6.5, 0.5])
    payload, scales = _quantize_tosa_reference(data, torch.float4_e2m1fn_x2, 32)
    assert scales.view(torch.uint8).tolist() == [127]
    assert payload.tolist() == [0x17] + [0] * 15
    actual = to_dtype(payload, scales, torch.float4_e2m1fn_x2, 32, torch.float32)
    torch.testing.assert_close(actual[:2], torch.tensor([6.0, 0.5]), rtol=0, atol=0)
    rceil_scale, rceil_payload = to_mx(
        data, torch.float4_e2m1fn_x2, 32, scaling_mode=ScaleCalculationMode.RCEIL
    )
    assert rceil_scale.view(torch.uint8).tolist() == [128]
    rceil = to_dtype(
        rceil_payload, rceil_scale, torch.float4_e2m1fn_x2, 32, torch.float32
    )
    torch.testing.assert_close(rceil[:2], torch.tensor([6.0, 0.0]), rtol=0, atol=0)
    legacy = _cast_to_block_scaled_cpu_ref(data, torch.float4_e2m1fn_x2, 32)
    torch.testing.assert_close(legacy, rceil, rtol=0, atol=0)


@pytest.mark.parametrize("conv2d", [False, True], ids=["linear", "conv2d"])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_frontend_preserves_legacy_activation_policy(conv2d, input_dtype):
    module = (
        torch.nn.Conv2d(32, 1, 1, bias=False)
        if conv2d
        else torch.nn.Linear(32, 1, bias=False)
    ).to(input_dtype)
    with torch.no_grad():
        module.weight.fill_(1)
    model = torch.nn.Sequential(module).eval()
    to_mxfp(model, MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2))
    data = torch.zeros((1, 32, 1, 1) if conv2d else (1, 32), dtype=input_dtype)
    data.flatten()[:2] = torch.tensor([6.5, 0.5])
    actual = model(data)
    torch.testing.assert_close(actual, torch.full_like(actual, 6.0), rtol=0, atol=0)
    assert actual.dtype == input_dtype


@pytest.mark.parametrize("shape", [(), (2, 0), (2, 31), (2, 33)])
def test_scaling_rejects_invalid_shape(shape):
    with pytest.raises(ValueError, match="block-aligned"):
        _quantize_tosa_reference(torch.empty(shape), torch.float4_e2m1fn_x2, 32)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.int8])
def test_scaling_rejects_unsupported_input_dtype(input_dtype):
    with pytest.raises(ValueError, match="FP32 or BF16"):
        _quantize_tosa_reference(
            torch.empty(32, dtype=input_dtype), torch.float4_e2m1fn_x2, 32
        )


@pytest.mark.parametrize(
    "output_dtype,block_size", [(torch.float32, 32), (torch.float4_e2m1fn_x2, 16)]
)
def test_scaling_rejects_unsupported_format_or_block_size(output_dtype, block_size):
    with pytest.raises(ValueError, match="MXFP format and block size 32"):
        _quantize_tosa_reference(torch.empty(32), output_dtype, block_size)
