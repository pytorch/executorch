# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import pytest
import torch
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.arm.tosa.tosa_mx_tensor import (
    BlockSize,
    MXFPValueType,
    ScaleType,
    TosaMxTensor,
    TosaMxType,
)
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


@pytest.mark.parametrize(
    "format_name", ["f4e2m1", "fp6e2m3", "fp6e3m2", "f8e4m3", "f8e5m2"]
)
def test_tosa_mx_tensor_fake_reconstruction(format_name):
    tensor_type = TosaMxType.from_payload_format(format_name)
    payload = torch.empty(
        2, 32 if format_name == "f4e2m1" else 64, dtype=tensor_type.payload_dtype
    )
    value = TosaMxTensor(
        payload, torch.empty(2, 2, dtype=torch.float8_e8m0fnu), tensor_type
    )

    with FakeTensorMode() as mode:
        fake_value = mode.from_tensor(value)

    assert isinstance(fake_value, TosaMxTensor)
    assert isinstance(fake_value.payload, FakeTensor)
    assert isinstance(fake_value.scales, FakeTensor)
    assert fake_value.mx_type == tensor_type
    assert fake_value.shape == value.shape == (2, 64)
    assert fake_value.stride() == value.stride() == (64, 1)
    assert fake_value.dtype == value.dtype == torch.float32
    assert fake_value.payload.dtype == tensor_type.payload_dtype


@pytest.mark.parametrize("fake", [False, True])
def test_tosa_mx_tensor_registered_clone(monkeypatch, fake):
    monkeypatch.setattr(TosaMxTensor, "_operator_impls", {})
    clone = torch.ops.aten.clone.default
    calls = []

    @TosaMxTensor.implements(clone)
    def clone_impl(func, types, args, kwargs):
        assert func is clone
        assert types == (TosaMxTensor,)
        assert kwargs == {"memory_format": torch.contiguous_format}
        calls.append(args[0])
        return TosaMxTensor(
            func(args[0].payload, **kwargs),
            func(args[0].scales, **kwargs),
            args[0].mx_type,
        )

    with FakeTensorMode() if fake else nullcontext():
        value = TosaMxTensor(
            torch.arange(64, dtype=torch.uint8).reshape(2, 32),
            torch.ones(2, 2).to(torch.float8_e8m0fnu),
            TosaMxType.from_payload_format("f4e2m1"),
        )
        result = torch.clone(value, memory_format=torch.contiguous_format)

        assert len(calls) == 1
        assert calls[0] is value
        assert isinstance(result, TosaMxTensor)
        assert result is not value
        assert result.mx_type == value.mx_type
        assert result.shape == value.shape
        assert result.dtype == value.dtype
        assert result.payload is not value.payload
        assert result.scales is not value.scales
        if fake:
            assert isinstance(result.payload, FakeTensor)
            assert isinstance(result.scales, FakeTensor)
        else:
            assert result.payload.data_ptr() != value.payload.data_ptr()
            assert result.scales.data_ptr() != value.scales.data_ptr()
            torch.testing.assert_close(result.payload, value.payload)
            torch.testing.assert_close(
                result.scales.view(torch.uint8), value.scales.view(torch.uint8)
            )


@pytest.mark.parametrize("operation", [lambda x: x + x, lambda x: x.transpose(0, 1)])
def test_tosa_mx_tensor_rejects_unhandled_operations(operation):
    value = TosaMxTensor(
        torch.empty(2, 32, dtype=torch.uint8),
        torch.empty(2, 2, dtype=torch.float8_e8m0fnu),
        TosaMxType.from_payload_format("f4e2m1"),
    )
    with pytest.raises(NotImplementedError, match="TosaMxTensor does not implement"):
        operation(value)


@pytest.mark.parametrize("scales_shape", [(2, 1), (4,)])
def test_tosa_mx_tensor_rejects_invalid_scale_geometry(scales_shape):
    with pytest.raises(ValueError, match="logical block geometry"):
        TosaMxTensor(
            torch.empty(2, 32, dtype=torch.uint8),
            torch.empty(scales_shape, dtype=torch.float8_e8m0fnu),
            TosaMxType.from_payload_format("f4e2m1"),
        )


@pytest.mark.parametrize(
    "format_name,value_type",
    [
        ("f4e2m1", MXFPValueType.FP4E2M1FN),
        ("fp6e2m3", MXFPValueType.FP6E2M3FN),
        ("fp6e3m2", MXFPValueType.FP6E3M2FN),
        ("f8e4m3", MXFPValueType.FP8E4M3FN),
        ("f8e5m2", MXFPValueType.FP8E5M2),
    ],
)
def test_mx_type(format_name, value_type):
    tensor_type = TosaMxType.from_payload_format(format_name)
    assert tensor_type.value_type is value_type
    assert tensor_type.scale_type is ScaleType.FP8UE8M0
    assert tensor_type.block_size is BlockSize.BLOCK_SIZE_32
    assert tensor_type.type_id == f"bs32_fp8ue8m0_{value_type.value}_t"
    assert TosaMxType.from_id(tensor_type.type_id) == tensor_type
    spec = TosaSpecification.create_from_string(
        f"TOSA-1.1+FP+mx_common+{tensor_type.format_extension}"
    )
    tensor_type.validate_spec(spec)
    for spec_string in [
        "TOSA-1.0+FP",
        "TOSA-1.1+FP+mxfp",
        "TOSA-1.1+FP+mx_common",
        "TOSA-1.1+INT",
    ]:
        with pytest.raises(ValueError, match="does not support"):
            tensor_type.validate_spec(TosaSpecification.create_from_string(spec_string))


@pytest.mark.parametrize("format_name", ["fp4", "fp8", "f8e4m3fn", ""])
def test_mx_type_rejects_unknown_payload_format(format_name):
    with pytest.raises(ValueError, match="Unsupported MXFP payload format"):
        TosaMxType.from_payload_format(format_name)


@pytest.mark.parametrize(
    "type_id",
    [
        "bs16_fp8ue8m0_fp4e2m1fn_t",
        "bs32_fp8ue5m3_fp4e2m1fn_t",
        "bs32_fp8ue8m0_fp4_t",
        "fp4e2m1fn",
    ],
)
def test_mx_type_rejects_unsupported_identifier(type_id):
    with pytest.raises(ValueError, match="Unsupported TOSA MX type identifier"):
        TosaMxType.from_id(type_id)


@pytest.mark.parametrize(
    "kwargs",
    [{"value_type": "fp4"}, {"scale_type": "fp8ue8m0"}, {"block_size": 32}],
)
def test_mx_type_rejects_invalid_components(kwargs):
    components = {"value_type": MXFPValueType.FP4E2M1FN, **kwargs}
    with pytest.raises(ValueError, match="Unsupported TOSA block-scale type") as error:
        TosaMxType(**components)
    for name, value in components.items():
        assert f"{name}={value!r}" in str(error.value)
    assert "Expected value_type to be an MXFPValueType member" in str(error.value)
    assert "scale_type=ScaleType.FP8UE8M0" in str(error.value)
    assert "block_size=BlockSize.BLOCK_SIZE_32" in str(error.value)


@pytest.mark.parametrize(
    "payload_shape,payload_dtype,scales_shape,scales_dtype,error",
    [
        ((2, 32), torch.int8, (2, 2), torch.float8_e8m0fnu, "storage dtype"),
        ((2, 32), torch.uint8, (2, 2), torch.uint8, "E8M0 scales"),
        ((), torch.uint8, (), torch.float8_e8m0fnu, "at least one dimension"),
        ((2, 0), torch.uint8, (2, 0), torch.float8_e8m0fnu, "block-aligned"),
        ((2, 17), torch.uint8, (2, 1), torch.float8_e8m0fnu, "block-aligned"),
    ],
)
def test_tosa_mx_tensor_rejects_invalid_storage(
    payload_shape, payload_dtype, scales_shape, scales_dtype, error
):
    with pytest.raises(ValueError, match=error):
        TosaMxTensor(
            torch.empty(payload_shape, dtype=payload_dtype),
            torch.empty(scales_shape, dtype=scales_dtype),
            TosaMxType.from_payload_format("f4e2m1"),
        )


@pytest.mark.parametrize("component", ["payload", "scales"])
def test_tosa_mx_tensor_rejects_noncontiguous_components(component):
    payload = torch.empty(2, 32, dtype=torch.uint8)
    scales = torch.empty(2, 2, dtype=torch.float8_e8m0fnu)
    if component == "payload":
        payload = torch.empty(32, 2, dtype=torch.uint8).t()
    else:
        scales = scales.t()
    with pytest.raises(ValueError, match="contiguous components"):
        TosaMxTensor(payload, scales, TosaMxType.from_payload_format("f4e2m1"))


def test_tosa_mx_tensor_rejects_mismatched_devices():
    with pytest.raises(ValueError, match="share a device"):
        TosaMxTensor(
            torch.empty(2, 32, dtype=torch.uint8),
            torch.empty(2, 2, dtype=torch.float8_e8m0fnu, device="meta"),
            TosaMxType.from_payload_format("f4e2m1"),
        )


@pytest.mark.parametrize(
    "outer_size,outer_stride,error",
    [
        ((4, 32), (32, 1), "logical shape changed"),
        ((2, 64), (1, 2), "logical strides changed"),
    ],
)
def test_tosa_mx_tensor_rejects_inconsistent_reconstruction(
    outer_size, outer_stride, error
):
    with pytest.raises(ValueError, match=error):
        TosaMxTensor.__tensor_unflatten__(
            {
                "payload": torch.empty(2, 32, dtype=torch.uint8),
                "scales": torch.empty(2, 2, dtype=torch.float8_e8m0fnu),
            },
            TosaMxType.from_payload_format("f4e2m1"),
            outer_size,
            outer_stride,
        )
