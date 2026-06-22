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
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2
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
    return isinstance(module, torch.nn.Linear)


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
