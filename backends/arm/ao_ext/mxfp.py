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
from torchao.prototype.mx_formats.mx_tensor import to_dtype, to_mx
from torchao.quantization import quantize_


def _match_supported_modules(module: torch.nn.Module, _name: str) -> bool:
    """Default filter function that matches supported modules."""
    return isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))


def _cast_to_block_scaled_cpu_ref(
    input: torch.Tensor,
    output_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    """Emulate the current TOSA activation cast in eager mode."""
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

    weight_dtype: torch.dtype = torch.float8_e4m3fn
    weight_scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL

    # Only block size of 32 is currently supported for now, so we hardcode it here.
    @property
    def block_size(self) -> int:
        return 32

    def __post_init__(self) -> None:
        if self.weight_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
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
