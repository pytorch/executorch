# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Represent TOSA composite MXFP tensors and their element types."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import torch
from executorch.backends.arm.tosa.specification import TosaSpecification


class BlockSize(Enum):
    """Block sizes supported by TOSA MXFP tensors."""

    BLOCK_SIZE_32 = 32


class ScaleType(Enum):
    """Scale element types supported by TOSA MXFP tensors."""

    FP8UE8M0 = "fp8ue8m0"


class MXFPValueType(Enum):
    """TOSA MX floating-point value element types."""

    FP4E2M1FN = "fp4e2m1fn"
    FP6E2M3FN = "fp6e2m3fn"
    FP6E3M2FN = "fp6e3m2fn"
    FP8E4M3FN = "fp8e4m3fn"
    FP8E5M2 = "fp8e5m2"


_PAYLOAD_TYPES = {
    "f4e2m1": MXFPValueType.FP4E2M1FN,
    "fp6e2m3": MXFPValueType.FP6E2M3FN,
    "fp6e3m2": MXFPValueType.FP6E3M2FN,
    "f8e4m3": MXFPValueType.FP8E4M3FN,
    "f8e5m2": MXFPValueType.FP8E5M2,
}


@dataclass(frozen=True)
class TosaMxType:
    """Represent a permitted TOSA MXFP block-scale element type."""

    value_type: MXFPValueType
    scale_type: ScaleType = ScaleType.FP8UE8M0
    block_size: BlockSize = BlockSize.BLOCK_SIZE_32

    def __post_init__(self) -> None:
        """Reject components outside the supported TOSA type combinations."""
        if (
            not isinstance(self.value_type, MXFPValueType)
            or self.scale_type is not ScaleType.FP8UE8M0
            or self.block_size is not BlockSize.BLOCK_SIZE_32
        ):
            raise ValueError(
                "Unsupported TOSA block-scale type combination: "
                f"value_type={self.value_type!r}, scale_type={self.scale_type!r}, "
                f"block_size={self.block_size!r}. "
                "Expected value_type to be an MXFPValueType member, "
                "scale_type=ScaleType.FP8UE8M0 and "
                "block_size=BlockSize.BLOCK_SIZE_32."
            )

    @property
    def type_id(self) -> str:
        """Return the concrete type name defined by the TOSA specification."""
        return (
            f"bs{self.block_size.value}_{self.scale_type.value}_"
            f"{self.value_type.value}_t"
        )

    @property
    def format_extension(self) -> str:
        """Return the extension required by the MX value format."""
        return f"mx_{self.value_type.value}"

    @property
    def payload_dtype(self) -> torch.dtype:
        """Return the physical PyTorch dtype of canonical payload storage."""
        if self.value_type is MXFPValueType.FP8E4M3FN:
            return torch.float8_e4m3fn
        if self.value_type is MXFPValueType.FP8E5M2:
            return torch.float8_e5m2
        # FP4 packs two values per byte; FP6 stores one value per byte.
        return torch.uint8

    @classmethod
    def from_payload_format(cls, payload_format: str) -> "TosaMxType":
        """Translate a frontend payload format into a TOSA type.

        Args:
            payload_format (str): MXFP format recorded by the frontend.

        Returns:
            TosaMxType: Corresponding TOSA type.

        """
        if payload_format not in _PAYLOAD_TYPES:
            raise ValueError(f"Unsupported MXFP payload format: {payload_format}")
        return cls(_PAYLOAD_TYPES[payload_format])

    @classmethod
    def from_id(cls, type_id: str) -> "TosaMxType":
        """Resolve an explicit internal operator type identifier.

        Args:
            type_id (str): Concrete TOSA block-scale type name.

        Returns:
            TosaMxType: Validated TOSA type.

        """
        for value_type in MXFPValueType:
            candidate = cls(value_type)
            if candidate.type_id == type_id:
                return candidate
        raise ValueError(f"Unsupported TOSA MX type identifier: {type_id}")

    def validate_spec(self, spec: TosaSpecification) -> None:
        """Check the target extensions required by this concrete type.

        Args:
            spec (TosaSpecification): Selected backend target.

        """
        if not spec.support_float() or not all(
            spec.support_extension(extension)
            for extension in ("mx_common", self.format_extension)
        ):
            raise ValueError(f"{spec} does not support {self.type_id}")


class TosaMxTensor(torch.Tensor):
    """Carry canonical MX payloads, scales and their logical TOSA type.

    The wrapper reports FP32; mx_type describes its MXFP encoding. Payloads and
    scales use contiguous canonical storage. Operations must explicitly handle
    both components; unsupported operations fail rather than treating the
    wrapper as an ordinary FP32 tensor.

    """

    payload: torch.Tensor
    scales: torch.Tensor
    mx_type: TosaMxType

    _operator_impls: ClassVar[dict] = {}
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        payload: torch.Tensor,
        scales: torch.Tensor,
        mx_type: TosaMxType,
    ):
        """Construct a logical tensor from canonical MXFP storage.

        Args:
            cls (type): Wrapper subclass to construct.
            payload (torch.Tensor): Encoded values; FP4 packs two per byte.
            scales (torch.Tensor): One E8M0 scale per block of 32 values.
            mx_type (TosaMxType): Complete MXFP element type.

        Returns:
            TosaMxTensor: FP32 wrapper with the logical, unpacked shape.

        """
        if not isinstance(mx_type, TosaMxType):
            raise TypeError("TosaMxTensor requires a TosaMxType")
        if payload.dtype != mx_type.payload_dtype:
            raise ValueError("Payload storage dtype disagrees with TOSA MX type")
        if scales.dtype != torch.float8_e8m0fnu:
            raise ValueError("TosaMxTensor requires E8M0 scales")
        if payload.device != scales.device:
            raise ValueError("Payload and scales must share a device")
        if not payload.is_contiguous() or not scales.is_contiguous():
            raise ValueError("TosaMxTensor requires contiguous components")
        shape = list(payload.shape)
        if not shape:
            raise ValueError("TosaMxTensor requires at least one dimension")
        if mx_type.value_type is MXFPValueType.FP4E2M1FN:
            shape[-1] *= 2
        block_size = mx_type.block_size.value
        if shape[-1] == 0 or shape[-1] % block_size:
            raise ValueError("TosaMxTensor requires a block-aligned final axis")
        if tuple(scales.shape) != (*shape[:-1], shape[-1] // block_size):
            raise ValueError("Scales do not match the logical block geometry")
        value = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=torch.float32,
            device=payload.device,
            requires_grad=False,
        )
        value.payload = payload
        value.scales = scales
        value.mx_type = mx_type
        return value

    def __tensor_flatten__(self):
        """Expose storage tensors and type metadata for tracing."""
        return ["payload", "scales"], self.mx_type

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        """Rebuild the wrapper, preserving its logical shape and strides."""
        value = cls(inner_tensors["payload"], inner_tensors["scales"], metadata)
        if outer_size is not None and tuple(outer_size) != tuple(value.shape):
            raise ValueError("TosaMxTensor logical shape changed during reconstruction")
        if outer_stride is not None and tuple(outer_stride) != value.stride():
            raise ValueError(
                "TosaMxTensor logical strides changed during reconstruction"
            )
        return value

    @classmethod
    def implements(cls, op):
        """Register an operation that explicitly supports composite values.

        Args:
            op (torch._ops.OpOverload): Operation handled by the implementation.

        Returns:
            Callable: Decorator registering the composite implementation.

        """

        def register(impl):
            cls._operator_impls[op] = impl
            return impl

        return register

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Dispatch only to implementations that handle composite storage."""
        impl = cls._operator_impls.get(func)
        if impl is None:
            raise NotImplementedError(f"TosaMxTensor does not implement {func}")
        return impl(func, types, args, kwargs or {})

    def __repr__(self):
        """Display the logical shape and complete MXFP type."""
        return (
            f"TosaMxTensor(shape={tuple(self.shape)}, "
            f"type={self.mx_type.type_id}, device={self.device})"
        )
