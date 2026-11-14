# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``triton::convolution``: the canonical convolution op of the CUDA backend.

The op body allocates the output once, with the size and stride of
``out_shape_and_stride``, and hands it to the first template whose
``supports()`` accepts the call; the template's schedules only write into it.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
import triton
from executorch.backends.cuda.triton.kernels.conv.core import direct
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    check_params,
    ConvCall,
    ConvParams,
    out_shape_and_stride,
)
from torch.library import triton_op


@dataclass(frozen=True)
class ConvTemplate:
    name: str
    supports: Callable[[ConvCall], Optional[str]]
    schedules: tuple[str, ...]
    schedule_supports: Callable[[str, ConvCall], Optional[str]]
    launch: Callable[..., None]
    run_schedule: Callable[..., None]
    # Why a call ``supports`` accepts is still left to the regular lowering
    # (outside the benchmarked range where the template wins), or None.
    worth_routing: Callable[[ConvCall], Optional[str]]


TEMPLATES = (
    ConvTemplate(
        "direct",
        direct.supports,
        tuple(s.name for s in direct.SCHEDULES),
        direct.schedule_supports,
        direct.launch,
        direct.run_schedule,
        direct.worth_routing,
    ),
)

_FORCED: contextvars.ContextVar[Optional[tuple[str, Optional[triton.Config]]]] = (
    contextvars.ContextVar("forced_conv_schedule", default=None)
)


@contextlib.contextmanager
def force_conv_candidate(
    schedule: str, config: Optional[triton.Config] = None
) -> Iterator[None]:
    """For tests: ``triton::convolution`` runs ``schedule`` without timing, with
    ``config`` instead of its autotuned config if given, and raises if the
    schedule cannot run the call."""
    token = _FORCED.set((schedule, config))
    try:
        yield
    finally:
        _FORCED.reset(token)


def select_template(call: ConvCall) -> ConvTemplate:
    reasons = []
    for template in TEMPLATES:
        reason = template.supports(call)
        if reason is None:
            return template
        reasons.append(reason)
    raise ValueError(f"no template supports {call.signature()}: {'; '.join(reasons)}")


def _canonical_call(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: list[int],
    padding_before: list[int],
    padding_after: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> ConvCall:
    params = ConvParams(
        tuple(stride),
        tuple(padding_before),
        tuple(padding_after),
        tuple(dilation),
        transposed,
        tuple(output_padding),
        groups,
    )
    check_params(params)
    if not input.dim() == weight.dim() == params.rank + 2:
        raise ValueError(
            f"{params.rank}-D convolution of a {input.dim()}-D input with a "
            f"{weight.dim()}-D weight"
        )
    return ConvCall(input, weight, bias, params)


@triton_op("triton::convolution", mutates_args={})
def convolution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: list[int],
    padding_before: list[int],
    padding_after: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    call = _canonical_call(
        input,
        weight,
        bias,
        stride,
        padding_before,
        padding_after,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    params = call.params
    template = select_template(call)
    size, stride_out = out_shape_and_stride(call)
    out = torch.empty_strided(size, stride_out, dtype=input.dtype, device=input.device)
    forced = _FORCED.get()
    if forced is None:
        template.launch(input, weight, bias, out, params)
        return out
    schedule, config = forced
    reason = template.schedule_supports(schedule, call)
    if reason is not None:
        raise ValueError(f"forced schedule cannot run {call.signature()}: {reason}")
    template.run_schedule(schedule, input, weight, bias, out, params, config)
    return out


@convolution.register_fake
def _convolution_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: list[int],
    padding_before: list[int],
    padding_after: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    call = _canonical_call(
        input,
        weight,
        bias,
        stride,
        padding_before,
        padding_after,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    size, stride_out = out_shape_and_stride(call)
    return input.new_empty_strided(size, stride_out)
