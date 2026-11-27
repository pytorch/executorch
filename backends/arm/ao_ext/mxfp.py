# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from executorch.exir._warnings import experimental
from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import (
    f32_to_f4_unpacked,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
    pack_uint4,
)
from torchao.prototype.mx_formats.mx_tensor import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    to_dtype,
    to_mx,
)
from torchao.quantization import quantize_


# Pytorch lacks dtypes for the FP6 types, so we use ao's string representations for those.
MXFPDType = torch.dtype | str


SUPPORTED_MXFP_DTYPES: set[MXFPDType] = {
    torch.float4_e2m1fn_x2,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    # Use ao's string representations.
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
}


_DTYPE_TO_STR: dict[MXFPDType, str] = {
    DTYPE_FP6_E2M3: "fp6e2m3",
    DTYPE_FP6_E3M2: "fp6e3m2",
    torch.float4_e2m1fn_x2: "f4e2m1",
    torch.float8_e4m3fn: "f8e4m3",
    torch.float8_e5m2: "f8e5m2",
}


_STR_TO_DTYPE = {value: key for (key, value) in _DTYPE_TO_STR.items()}


def mxfp_dtype_to_str(dtype: MXFPDType) -> str:
    try:
        return _DTYPE_TO_STR[dtype]
    except KeyError as e:
        supported = ", ".join(str(dtype) for dtype in _DTYPE_TO_STR)
        raise ValueError(
            f"Unsupported MXFP dtype {dtype}. Supported dtypes: {supported}"
        ) from e


def mxfp_str_to_dtype(dtype: str) -> MXFPDType:
    try:
        return _STR_TO_DTYPE[dtype]
    except KeyError as e:
        supported = ", ".join(sorted(_STR_TO_DTYPE))
        raise ValueError(
            f"Unsupported MXFP dtype string {dtype!r}. Supported strings: {supported}"
        ) from e


def _match_supported_modules(module: torch.nn.Module, _name: str) -> bool:
    """Default filter function that matches supported modules."""
    return isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))


# Upper rounding boundaries used by TOSA 1.1 calc_block_scale. The E4M3
# boundary is inclusive; the other formats round away from their finite maximum.
_ROUNDING_LIMITS: dict[MXFPDType, float] = {
    torch.float4_e2m1fn_x2: 7.0,
    DTYPE_FP6_E2M3: 7.75,
    DTYPE_FP6_E3M2: 30.0,
    torch.float8_e4m3fn: 464.0,
    torch.float8_e5m2: 61440.0,
}


def _choose_tosa_scale(
    input: torch.Tensor, output_dtype: MXFPDType, block_size: int
) -> torch.Tensor:
    """Choose E8M0 block scales using TOSA 1.1 reference scaling.

    Args:
        input (torch.Tensor): FP32 or BF16 values blocked along the final axis.
        output_dtype (MXFPDType): MXFP payload format.
        block_size (int): Number of elements per block, currently 32.

    Returns:
        torch.Tensor: One E8M0 scale per block, with the final axis divided
            by the block size.

    """
    if output_dtype not in _ROUNDING_LIMITS or block_size != 32:
        raise ValueError(
            "TOSA-reference scaling requires an MXFP format and block size 32"
        )
    if input.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("TOSA-reference scaling requires FP32 or BF16 input")
    if input.ndim == 0 or input.shape[-1] == 0 or input.shape[-1] % block_size:
        raise ValueError("TOSA-reference scaling requires block-aligned input")
    shape = (*input.shape[:-1], input.shape[-1] // block_size)
    maximum = input.double().reshape(*shape, block_size).abs().amax(dim=-1)
    valid = torch.isfinite(maximum) & (maximum != 0)
    safe_maximum = torch.where(valid, maximum, torch.ones_like(maximum))
    logarithm = torch.log2(safe_maximum / _ROUNDING_LIMITS[output_dtype])
    exponent = (
        logarithm.ceil()
        if output_dtype == torch.float8_e4m3fn
        else logarithm.floor() + 1
    )
    encoded = (exponent.clamp(-127, 127) + 127).to(torch.uint8)
    encoded = torch.where(maximum == 0, torch.zeros_like(encoded), encoded)
    encoded = torch.where(torch.isfinite(maximum), encoded, 255)
    return encoded.contiguous().view(torch.float8_e8m0fnu)


def _quantize_tosa_reference(
    input: torch.Tensor, output_dtype: MXFPDType, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert floating-point values to MXFP payloads and E8M0 scales.

    Args:
        input (torch.Tensor): FP32 or BF16 values blocked along the final axis.
        output_dtype (MXFPDType): MXFP payload format.
        block_size (int): Number of elements per block, currently 32.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Payload storage and E8M0 scales.

    """
    scale = _choose_tosa_scale(input, output_dtype, block_size)
    blocks = input.double().reshape(*scale.shape, block_size)
    scaled = (
        (blocks / scale.double().unsqueeze(-1))
        .float()
        .reshape(input.shape)
        .contiguous()
    )
    if output_dtype == torch.float4_e2m1fn_x2:
        payload = pack_uint4(f32_to_f4_unpacked(scaled))
    elif output_dtype == DTYPE_FP6_E2M3:
        payload = f32_to_f6_e2m3_unpacked(scaled)
    elif output_dtype == DTYPE_FP6_E3M2:
        payload = f32_to_f6_e3m2_unpacked(scaled)
    else:
        payload = scaled.to(output_dtype)
    return payload, scale


def _cast_to_block_scaled_cpu_ref(
    input: torch.Tensor,
    output_dtype: MXFPDType,
    block_size: int,
) -> torch.Tensor:
    """Emulate legacy TOSA activation conversion using TorchAO RCEIL."""
    input_scale, input_qdata = to_mx(
        input.to(torch.float32).contiguous(),
        elem_dtype=output_dtype,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )
    return to_dtype(
        input_qdata,
        input_scale,
        output_dtype,
        block_size,
        torch.float32,
    )


@experimental("This API is experimental and may change without notice.")
@dataclass
class MXFPOpConfig(AOBaseConfig):
    """Configuration for Arm MXFP source transforms."""

    weight_dtype: MXFPDType = torch.float8_e4m3fn
    weight_scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL

    # Only block size of 32 is currently supported for now, so we hardcode it here.
    @property
    def block_size(self) -> int:
        return 32

    def __post_init__(self) -> None:
        if self.weight_dtype not in SUPPORTED_MXFP_DTYPES:
            raise ValueError(f"Unsupported weight_dtype: {self.weight_dtype}")
        if not isinstance(self.weight_scaling_mode, ScaleCalculationMode):
            raise ValueError(
                f"Unsupported weight_scaling_mode: {self.weight_scaling_mode}"
            )


@experimental("This API is experimental and may change without notice.")
def to_mxfp(
    model: torch.nn.Module,
    config: MXFPOpConfig,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
) -> None:
    """Convert matching modules in ``model`` to Arm MXFP modules in-place.

    Args:
        model (torch.nn.Module): Module to transform. Matching submodules are
            replaced in-place.
        config (MXFPOpConfig): Configuration controlling the MXFP conversion
            behavior.
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): Optional
            predicate that receives a module and its fully qualified name. When
            omitted, all modules supported by the MXFP transform are matched.

    """
    if filter_fn is None:
        filter_fn = _match_supported_modules

    quantize_(model, config, filter_fn)
