# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed CMSIS-NN bindings for Cortex-M export."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, cast, ClassVar, Sequence, TYPE_CHECKING

import torch

_cmsis_nn: ModuleType | None = None
_cmsis_nn_module_candidates = (
    "executorch.backends.cortex_m.library._cmsis_nn.cmsis_nn",
    "cmsis_nn",
)


class _EnumValue:
    def __init__(self, enum_name: str, name: str, value: int) -> None:
        self._enum_name = enum_name
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"<{self._enum_name}.{self.name}: {self.value}>"

    def __str__(self) -> str:
        return f"{self._enum_name}.{self.name}"


class Backend:
    MVE: ClassVar[Backend]
    DSP: ClassVar[Backend]
    SCALAR: ClassVar[Backend]

    name: str
    value: int


Backend.MVE = cast(Backend, _EnumValue("Backend", "MVE", 0))
Backend.DSP = cast(Backend, _EnumValue("Backend", "DSP", 1))
Backend.SCALAR = cast(Backend, _EnumValue("Backend", "SCALAR", 2))


class CortexM:
    M0: ClassVar[CortexM]
    M0PLUS: ClassVar[CortexM]
    M3: ClassVar[CortexM]
    M4: ClassVar[CortexM]
    M7: ClassVar[CortexM]
    M23: ClassVar[CortexM]
    M33: ClassVar[CortexM]
    M35P: ClassVar[CortexM]
    M55: ClassVar[CortexM]
    M85: ClassVar[CortexM]

    name: str
    value: int


CortexM.M0 = cast(CortexM, _EnumValue("CortexM", "M0", 0))
CortexM.M0PLUS = cast(CortexM, _EnumValue("CortexM", "M0PLUS", 1))
CortexM.M3 = cast(CortexM, _EnumValue("CortexM", "M3", 2))
CortexM.M4 = cast(CortexM, _EnumValue("CortexM", "M4", 3))
CortexM.M7 = cast(CortexM, _EnumValue("CortexM", "M7", 4))
CortexM.M23 = cast(CortexM, _EnumValue("CortexM", "M23", 5))
CortexM.M33 = cast(CortexM, _EnumValue("CortexM", "M33", 6))
CortexM.M35P = cast(CortexM, _EnumValue("CortexM", "M35P", 7))
CortexM.M55 = cast(CortexM, _EnumValue("CortexM", "M55", 8))
CortexM.M85 = cast(CortexM, _EnumValue("CortexM", "M85", 9))


class DataType:
    A8W4: ClassVar[DataType]
    A8W8: ClassVar[DataType]
    A16W8: ClassVar[DataType]

    name: str
    value: int


DataType.A8W4 = cast(DataType, _EnumValue("DataType", "A8W4", 0))
DataType.A8W8 = cast(DataType, _EnumValue("DataType", "A8W8", 1))
DataType.A16W8 = cast(DataType, _EnumValue("DataType", "A16W8", 2))


if not TYPE_CHECKING:
    # First try to load cmsis_nn from executorch build, then external.
    # Load in such a way that we crash only if a cmsis_nn function is required.
    for module_name in _cmsis_nn_module_candidates:
        try:
            _real_cmsis_nn = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name not in module_name:
                raise exc
        else:
            _cmsis_nn = _real_cmsis_nn
            Backend = _real_cmsis_nn.Backend
            CortexM = _real_cmsis_nn.CortexM
            DataType = _real_cmsis_nn.DataType
            break


def _require_cmsis_nn() -> ModuleType:
    if _cmsis_nn is None:
        raise ModuleNotFoundError(
            f"Cortex-M backend dependencies are not installed (tried {','.join(_cmsis_nn_module_candidates)}). "
            "Build using install_executorch.sh, or pip install from the CMSIS-NN repo."
        )
    return _cmsis_nn


def resolve_backend(cpu: CortexM) -> Backend:
    return _require_cmsis_nn().resolve_backend(cpu)


def convolve_wrapper_buffer_size(
    backend: Backend,
    data_type: DataType,
    *,
    input_nhwc: Sequence[int],
    filter_nhwc: Sequence[int],
    output_nhwc: Sequence[int],
    padding_hw: Sequence[int],
    stride_hw: Sequence[int],
    dilation_hw: Sequence[int],
    input_offset: int = 0,
    output_offset: int = 0,
    activation_min: int = -128,
    activation_max: int = 127,
    weights_packed: bool = False,
) -> int:
    bindings = _require_cmsis_nn()
    return bindings.convolve_wrapper_buffer_size(
        backend,
        data_type,
        input_nhwc=input_nhwc,
        filter_nhwc=filter_nhwc,
        output_nhwc=output_nhwc,
        padding_hw=padding_hw,
        stride_hw=stride_hw,
        dilation_hw=dilation_hw,
        input_offset=input_offset,
        output_offset=output_offset,
        activation_min=activation_min,
        activation_max=activation_max,
        filter_format=(
            bindings.FilterFormat.PACKED_4X16
            if weights_packed
            else bindings.FilterFormat.ROW_MAJOR
        ),
    )


def depthwise_conv_wrapper_buffer_size(
    backend: Backend,
    data_type: DataType,
    *,
    input_nhwc: Sequence[int],
    filter_nhwc: Sequence[int],
    output_nhwc: Sequence[int],
    padding_hw: Sequence[int],
    stride_hw: Sequence[int],
    dilation_hw: Sequence[int],
    ch_mult: int,
    input_offset: int = 0,
    output_offset: int = 0,
    activation_min: int = -128,
    activation_max: int = 127,
) -> int:
    return _require_cmsis_nn().depthwise_conv_wrapper_buffer_size(
        backend,
        data_type,
        input_nhwc=input_nhwc,
        filter_nhwc=filter_nhwc,
        output_nhwc=output_nhwc,
        padding_hw=padding_hw,
        stride_hw=stride_hw,
        dilation_hw=dilation_hw,
        ch_mult=ch_mult,
        input_offset=input_offset,
        output_offset=output_offset,
        activation_min=activation_min,
        activation_max=activation_max,
    )


def fully_connected_buffer_size(
    backend: Backend,
    data_type: DataType,
    *,
    filter_nhwc: Sequence[int],
) -> int:
    return _require_cmsis_nn().fully_connected_buffer_size(
        backend,
        data_type,
        filter_nhwc=filter_nhwc,
    )


def vector_sum_s8(
    weights: torch.Tensor,
    lhs_offset: int,
    bias: torch.Tensor | None = None,
    rhs_offset: int = 0,
) -> torch.Tensor:
    result = _require_cmsis_nn().vector_sum_s8(
        weights.detach().cpu().contiguous().numpy(),
        lhs_offset,
        None if bias is None else bias.detach().cpu().contiguous().numpy(),
        rhs_offset,
    )
    return torch.from_numpy(result)


def is_aot_enabled() -> bool:
    return _require_cmsis_nn().aot_enabled


def convolve_s8_prepare_weights(
    weights: torch.Tensor,
    input_offset: int,
    bias: torch.Tensor | None = None,
    groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack int8 OHWI weights and precompute effective bias at export.

    Return a flat PACKED_4X16 weight buffer and an int32 effective-bias tensor.
    MVE convolution consumes these together at runtime without unpacking.
    """
    packed, effective_bias = _require_cmsis_nn().convolve_s8_prepare_weights(
        weights.detach().cpu().contiguous().numpy(),
        input_offset,
        None if bias is None else bias.detach().cpu().contiguous().numpy(),
        groups=groups,
    )
    return torch.from_numpy(packed).clone(), torch.from_numpy(effective_bias).clone()


def transpose_conv_buffer_size(
    backend: Backend,
    data_type: DataType,
    *,
    input_nhwc: Sequence[int],
    filter_nhwc: Sequence[int],
    output_nhwc: Sequence[int],
    padding_hw: Sequence[int],
    stride_hw: Sequence[int],
    dilation_hw: Sequence[int],
    padding_offsets_hw: Sequence[int] = (0, 0),
    input_offset: int = 0,
    output_offset: int = 0,
    activation_min: int = -128,
    activation_max: int = 127,
) -> int:
    return _require_cmsis_nn().transpose_conv_buffer_size(
        backend,
        data_type,
        input_nhwc=input_nhwc,
        filter_nhwc=filter_nhwc,
        output_nhwc=output_nhwc,
        padding_hw=padding_hw,
        stride_hw=stride_hw,
        dilation_hw=dilation_hw,
        padding_offsets_hw=padding_offsets_hw,
        input_offset=input_offset,
        output_offset=output_offset,
        activation_min=activation_min,
        activation_max=activation_max,
    )


def transpose_conv_reverse_conv_buffer_size(
    backend: Backend,
    data_type: DataType,
    *,
    input_nhwc: Sequence[int],
    filter_nhwc: Sequence[int],
    padding_hw: Sequence[int],
    stride_hw: Sequence[int],
    dilation_hw: Sequence[int] = (1, 1),
    padding_offsets_hw: Sequence[int] = (0, 0),
    input_offset: int = 0,
    output_offset: int = 0,
    activation_min: int = -128,
    activation_max: int = 127,
) -> int:
    return _require_cmsis_nn().transpose_conv_reverse_conv_buffer_size(
        backend,
        data_type,
        input_nhwc=input_nhwc,
        filter_nhwc=filter_nhwc,
        padding_hw=padding_hw,
        stride_hw=stride_hw,
        dilation_hw=dilation_hw,
        padding_offsets_hw=padding_offsets_hw,
        input_offset=input_offset,
        output_offset=output_offset,
        activation_min=activation_min,
        activation_max=activation_max,
    )


def avgpool_buffer_size(
    backend: Backend,
    data_type: DataType,
    *,
    dim_dst_width: int,
    ch_src: int,
) -> int:
    return _require_cmsis_nn().avgpool_buffer_size(
        backend,
        data_type,
        dim_dst_width=dim_dst_width,
        ch_src=ch_src,
    )


def __getattr__(name: str) -> Any:
    return getattr(_require_cmsis_nn(), name)


def __dir__() -> list[str]:
    cmsis_names = set() if _cmsis_nn is None else set(dir(_cmsis_nn))
    return sorted(set(globals()) | cmsis_names)
