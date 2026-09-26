# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import fx
from torchao.quantization.granularity import PerBlock
from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e._affine_quantization import AffineQuantizedMinMaxObserver
from torchao.quantization.pt2e.quantizer import QuantizationConfig, QuantizationSpec
from torchao.quantization.quant_primitives import MappingType


def make_per_tensor_qspec(
    dtype: torch.dtype,
    quant_min: int,
    quant_max: int,
    *,
    histogram_observer: bool = False,
    symmetric: bool = False,
    is_dynamic: bool = False,
    eps: float | None = None,
) -> QuantizationSpec:
    """Build a per-tensor ``QuantizationSpec`` (one scale/zero-point per tensor).

    Args:
        symmetric: When True the qscheme is ``per_tensor_symmetric`` (zero-point
            pinned to 0); otherwise ``per_tensor_affine`` (asymmetric).
        histogram_observer: When True use ``HistogramObserver``; otherwise use ``MinMaxObserver``.
        eps: optional floor on the computed scale (forwarded to the observer);
            leave None to use the observer's default. Activation observers want an
            explicit eps=2**-12: with a smaller floor a small-range activation can
            produce a tiny scale, which makes affine-transform parameter extraction
            infeasible on backends that approximate activations with a lookup table
            (e.g. HardSigmoid).
    """
    base_observer = HistogramObserver if histogram_observer else MinMaxObserver
    observer_ctr = (
        base_observer.with_args(eps=eps) if eps is not None else base_observer
    )
    return QuantizationSpec(
        dtype=dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=(torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine),
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=observer_ctr,
    )


def make_per_channel_qspec(
    dtype: torch.dtype,
    quant_min: int,
    quant_max: int,
    *,
    ch_axis: int = 0,
    eps: float = 2**-12,
    symmetric: bool = False,
    is_dynamic: bool = False,
) -> QuantizationSpec:
    """Build a per-channel ``QuantizationSpec`` (one scale/zero-point per slice
    along ``ch_axis``; axis 0 is the output-channel axis for conv/linear weights).

    Args:
        symmetric: When True the qscheme is ``per_channel_symmetric``; otherwise
            ``per_channel_affine`` (asymmetric).
    """
    return QuantizationSpec(
        dtype=dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=(
            torch.per_channel_symmetric if symmetric else torch.per_channel_affine
        ),
        ch_axis=ch_axis,
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=eps),
    )


def make_blockwise_qspec(
    dtype: torch.dtype,
    quant_min: int,
    quant_max: int,
    *,
    block_size: tuple[int, ...],
    symmetric: bool = True,
    eps: float = 2**-12,
) -> QuantizationSpec:
    """Build a blockwise ``QuantizationSpec`` whose ``convert`` emits
    ``torchao.quantize_affine`` / ``dequantize_affine`` nodes with an explicit
    ``block_size``.

    ``block_size`` is right-aligned to the tensor's trailing dims (leading dims
    pad with 1), so axis ``i`` gets ``shape[i] // block_size[i]`` scales -- i.e.
    arbitrary rectangular blocks, not just a 1-D group. The common per-row group
    case is ``block_size=(1, group_size)`` (one scale per ``group_size`` elements
    along the last dim), giving a 2-D weight a ``[rows, cols // group_size]``
    scale; ``block_size=(2, 4)`` instead tiles a ``[R, C]`` weight into
    ``[R//2, C//4]`` blocks. Each block dim must divide the tensor dim.

    Unlike the per-tensor / per-channel specs this carries no ``qscheme`` /
    ``ch_axis`` (blockwise has no torch.qscheme): the upstream
    ``AffineQuantizedMinMaxObserver`` owns node insertion via its ``convert`` hook
    (which ``convert_pt2e`` calls when an observer defines one), driven by the
    ``PerBlock`` granularity bound here. ``dtype`` is the integer container (e.g.
    int8 with quant_min/max -8/7 for W4).
    """
    mapping_type = MappingType.SYMMETRIC if symmetric else MappingType.ASYMMETRIC
    return QuantizationSpec(
        dtype=dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=None,
        ch_axis=None,
        is_dynamic=False,
        observer_or_fake_quant_ctr=AffineQuantizedMinMaxObserver.with_args(
            mapping_type=mapping_type,
            target_dtype=dtype,
            granularity=PerBlock(block_size=tuple(block_size)),
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        ),
    )


def make_qconfig(
    *,
    weight_per_channel: bool,
    act_dtype: torch.dtype = torch.int8,
    weight_dtype: torch.dtype = torch.int8,
    act_symmetric: bool = False,
    weight_symmetric: bool = False,
    act_qmin: Optional[int] = None,
    act_qmax: Optional[int] = None,
    weight_qmin: Optional[int] = None,
    weight_qmax: Optional[int] = None,
) -> QuantizationConfig:
    """Build a default quant config.

    Activations are per-tensor histogram (the same qspec instance is shared for
    input and output); the weight qspec is per-channel along axis 0 when
    ``weight_per_channel`` is set, otherwise per-tensor. Activation and weight
    dtypes / symmetry are independent. Explicit quantization bounds are used
    verbatim (e.g. -8/7 to pack W4 into int8). By default, signed symmetric
    weights use a balanced range (e.g. -127/127 for int8); all other defaults
    use the dtype's full range.
    """
    act_info = torch.iinfo(act_dtype)
    weight_info = torch.iinfo(weight_dtype)
    act_qmin = act_qmin if act_qmin is not None else act_info.min
    act_qmax = act_qmax if act_qmax is not None else act_info.max
    weight_qmax = weight_qmax if weight_qmax is not None else weight_info.max
    weight_qmin = (
        weight_qmin
        if weight_qmin is not None
        else (
            -weight_qmax
            if weight_symmetric and weight_info.min < 0
            else weight_info.min
        )
    )
    # eps=2**-12 floors the activation scale; without it a small-range activation
    # gets a tiny scale that breaks affine-transform parameter extraction on
    # backends that approximate activations with a lookup table (e.g. HardSigmoid
    # in mobilenetv3).
    act_qspec = make_per_tensor_qspec(
        act_dtype,
        act_qmin,
        act_qmax,
        histogram_observer=True,
        symmetric=act_symmetric,
        eps=2**-12,
    )
    weight_qspec = (
        make_per_channel_qspec(
            weight_dtype, weight_qmin, weight_qmax, symmetric=weight_symmetric
        )
        if weight_per_channel
        else make_per_tensor_qspec(
            weight_dtype, weight_qmin, weight_qmax, symmetric=weight_symmetric
        )
    )
    return QuantizationConfig(act_qspec, act_qspec, weight_qspec, None)


def arg1_is_scalar(node: fx.Node) -> bool:
    """A node_filter for the .Scalar binary variants: True when the second
    operand is a Python scalar (e.g. x + eps) rather than a tensor."""
    return not isinstance(node.args[1], fx.Node)
