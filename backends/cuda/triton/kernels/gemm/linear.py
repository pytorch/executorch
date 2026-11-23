# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``triton::linear``: ``x @ weight.T + bias`` in strict FP32.

``x`` is ``[*, K]``, ``weight`` ``[N, K]`` and ``bias`` ``[N]`` or None; the
output is ``[*, N]``. The op body allocates the output once, with
``output_strides`` (contiguous if None), and hands it to a schedule, which
only writes into it.

Layouts. The kernels address ``x`` and the output as matrices of ``M`` rows
(the product of the leading dims), each with one row stride and one column
stride, so their leading dims must flatten: for every pair of consecutive
leading dims ``i < j`` of size > 1, ``stride[i] == stride[j] * size[j]``.
Storage offsets are allowed. The output must also be a dense,
non-overlapping layout. ``weight`` is addressed with its two
strides. Everything else is refused with a reason (``layout_reason``).
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch
import triton
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_param,
    InvalidLaunchParam,
)
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    dense_strides,
    Schedule,
)
from executorch.backends.cuda.triton.kernels.gemm import tile
from executorch.backends.cuda.triton.routing import device_supports
from torch.library import triton_op

SCHEDULES = (Schedule("ieee_tile", 0),)
# The smallest tile any schedule launches, which bounds the grid.
_MIN_TILE = (16, 32)
_MAX_GRID = 2**31 - 1


def matmul_allows_tf32() -> bool:
    """Whether FP32 matrix multiplications may compute in TF32: the setting
    Inductor's Triton GEMM templates follow."""
    from torch._dynamo.device_interface import get_interface_for_device

    interface = get_interface_for_device("cuda")
    # Older PyTorch releases read the CUDA matmul setting directly.
    if not hasattr(interface, "allow_tf32"):
        return torch.backends.cuda.matmul.fp32_precision == "tf32"
    return bool(interface.allow_tf32())


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides, running = [], 1
    for size in reversed(shape):
        strides.append(running)
        running *= max(size, 1)
    return tuple(reversed(strides))


def row_stride(shape: Sequence[int], strides: Sequence[int]) -> Optional[int]:
    """The stride of the rows when the leading dims (all but the last) are
    flattened into one, or None if they do not flatten. 0 when there is at
    most one row."""
    lead = [(n, s) for n, s in zip(shape[:-1], strides[:-1]) if n != 1]
    if any(n == 0 for n, _ in lead):
        return 0
    for (_, outer), (inner_size, inner) in zip(lead, lead[1:]):
        if outer != inner * inner_size:
            return None
    return lead[-1][1] if lead else 0


@dataclass(frozen=True)
class LinearLayout:
    """The matrix view the kernels address: ``M`` rows of ``x`` and of the
    output, ``N`` output columns, ``K`` reduction steps, and their strides."""

    m: int
    n: int
    k: int
    x: tuple[int, int]
    weight: tuple[int, int]
    out: tuple[int, int]


def output_shape(x: torch.Tensor, weight: torch.Tensor) -> tuple[int, ...]:
    return (*x.shape[:-1], weight.shape[0])


def describe(t: Optional[torch.Tensor]) -> str:
    if t is None:
        return "None"
    dtype = str(t.dtype).removeprefix("torch.")
    offset = f"+{t.storage_offset()}" if t.storage_offset() else ""
    return f"{dtype}{list(t.shape)}@{list(t.stride())}{offset}"


def signature(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_strides: Sequence[int],
) -> str:
    return (
        f"linear(x={describe(x)}, weight={describe(weight)}, "
        f"bias={describe(bias)}, output_strides={list(output_strides)})"
    )


def layout_reason(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_strides: Sequence[int],
) -> Optional[str]:
    """Why ``triton::linear`` cannot run this call, or None."""
    if x.dim() < 1 or weight.dim() != 2:
        return f"a {x.dim()}-D input with a {weight.dim()}-D weight"
    if x.shape[-1] != weight.shape[1]:
        return f"input {list(x.shape)} and weight {list(weight.shape)} differ in K"
    tensors = [t for t in (x, weight, bias) if t is not None]
    if any(t.dtype != torch.float32 for t in tensors):
        return "dtypes " + ", ".join(str(t.dtype) for t in tensors) + ", needs float32"
    if any(t.device != x.device for t in tensors) or x.device.type != "cuda":
        return "tensors not all on one CUDA device"
    shape = output_shape(x, weight)
    if math.prod(shape) == 0 or x.shape[-1] == 0:
        return f"empty input {list(x.shape)} or weight {list(weight.shape)}"
    if bias is not None and (
        tuple(bias.shape) != (weight.shape[0],)
        or (bias.stride(0) != 1 and weight.shape[0] != 1)
    ):
        return f"bias {describe(bias)} is not a contiguous [{weight.shape[0]}] tensor"
    if row_stride(x.shape, x.stride()) is None:
        return f"the leading dims of input {describe(x)} do not flatten into rows"
    if len(output_strides) != len(shape) or not dense_strides(shape, output_strides):
        return f"output strides {list(output_strides)} are not a dense layout of {list(shape)}"
    if row_stride(shape, output_strides) is None:
        return f"the leading dims of output strides {list(output_strides)} do not flatten into rows"
    m = math.prod(shape[:-1])
    tiles = triton.cdiv(m, _MIN_TILE[0]) * triton.cdiv(shape[-1], _MIN_TILE[1])
    if tiles > _MAX_GRID:
        return f"{tiles} output tiles exceed the launch grid"
    return None


def linear_layout(
    x: torch.Tensor, weight: torch.Tensor, output_strides: Sequence[int]
) -> LinearLayout:
    """The matrix view of a call ``layout_reason`` accepts."""
    shape = output_shape(x, weight)
    return LinearLayout(
        math.prod(shape[:-1]),
        weight.shape[0],
        weight.shape[1],
        (row_stride(x.shape, x.stride()), x.stride(-1)),
        (weight.stride(0), weight.stride(1)),
        (row_stride(shape, output_strides), output_strides[-1]),
    )


_FORCED: contextvars.ContextVar[Optional[tuple[str, Optional[triton.Config]]]] = (
    contextvars.ContextVar("forced_linear_schedule", default=None)
)


@contextlib.contextmanager
def force_linear_candidate(
    schedule: str, config: Optional[triton.Config] = None
) -> Iterator[None]:
    """For tests and benchmarks: ``triton::linear`` runs ``schedule`` without
    timing, with ``config`` instead of its autotuned config if given, and
    raises if the schedule cannot run the call."""
    token = _FORCED.set((schedule, config))
    try:
        yield
    finally:
        _FORCED.reset(token)


def schedule_supports(name: str, x: torch.Tensor) -> Optional[str]:
    """Why schedule ``name`` cannot run a call ``layout_reason`` accepts, or
    None."""
    schedule = next((s for s in SCHEDULES if s.name == name), None)
    if schedule is None:
        return f"unknown schedule {name}"
    if not device_supports(schedule.min_capability, x.device):
        return f"{name} needs sm_{schedule.min_capability}"
    return None


def run_schedule(
    name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    layout: LinearLayout,
    config: Optional[triton.Config] = None,
) -> None:
    """Launches schedule ``name``; ``config`` replaces its autotuned config."""
    tile.run(x, weight, bias, out, layout, config)


@autotune_launch_param("schedule", tuple(s.name for s in SCHEDULES))
def launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    layout: LinearLayout,
    *,
    schedule: str,
) -> None:
    """Runs the linear into ``out``; during AOTI compile the schedule is timed."""
    reason = schedule_supports(schedule, x)
    if reason is not None:
        raise InvalidLaunchParam(reason)
    run_schedule(schedule, x, weight, bias, out, layout)


def _checked(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_strides: Optional[list[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = output_shape(x, weight) if weight.dim() == 2 else ()
    strides = (
        _contiguous_strides(shape) if output_strides is None else tuple(output_strides)
    )
    reason = layout_reason(x, weight, bias, strides)
    if reason is not None:
        raise ValueError(
            f"triton::linear cannot run {signature(x, weight, bias, strides)}: {reason}"
        )
    return shape, strides


@triton_op("triton::linear", mutates_args={})
def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_strides: Optional[list[int]] = None,
) -> torch.Tensor:
    shape, strides = _checked(x, weight, bias, output_strides)
    out = torch.empty_strided(shape, strides, dtype=x.dtype, device=x.device)
    layout = linear_layout(x, weight, strides)
    forced = _FORCED.get()
    if forced is None:
        launch(x, weight, bias, out, layout)
        return out
    schedule, config = forced
    reason = schedule_supports(schedule, x)
    if reason is not None:
        raise ValueError(
            f"forced schedule cannot run {signature(x, weight, bias, strides)}: {reason}"
        )
    run_schedule(schedule, x, weight, bias, out, layout, config)
    return out


@linear.register_fake
def _linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_strides: Optional[list[int]] = None,
) -> torch.Tensor:
    shape, strides = _checked(x, weight, bias, output_strides)
    return x.new_empty_strided(shape, strides)
