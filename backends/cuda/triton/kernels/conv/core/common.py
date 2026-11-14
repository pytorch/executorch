# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared pieces of ``triton::convolution``: the canonical arguments, binding
every ATen conv overload to them, and the output layout.

Canonical arguments. Every spatial argument has one entry per spatial dim.

- Forward (``transposed=False``): the input is padded with ``padding_before[i]``
  zeros in front of and ``padding_after[i]`` zeros behind spatial dim ``i``,
  then convolved with no padding:
  ``out = (in + before + after - dilation * (k - 1) - 1) // stride + 1``.
  The weight is ``[C_out, C_in / groups, *k]`` and ``output_padding`` is 0.
- Transposed (``transposed=True``): the full transposed convolution of length
  ``(in - 1) * stride + dilation * (k - 1) + 1`` gets ``output_padding[i]``
  positions appended, then ``before`` positions are cropped from the front and
  ``after`` from the back:
  ``out = (in - 1) * stride - before - after + dilation * (k - 1) + output_padding + 1``.
  The weight is ``[C_in, C_out / groups, *k]``. ATen's transposed overloads
  always bind ``before == after``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# (schema name, overload) -> spatial rank, or None when any rank is accepted.
CONV_OVERLOADS = {
    ("aten::conv1d", ""): 1,
    ("aten::conv2d", ""): 2,
    ("aten::conv3d", ""): 3,
    ("aten::conv1d", "padding"): 1,
    ("aten::conv2d", "padding"): 2,
    ("aten::conv3d", "padding"): 3,
    ("aten::conv_transpose1d", ""): 1,
    ("aten::conv_transpose2d", "input"): 2,
    ("aten::conv_transpose3d", "input"): 3,
    ("aten::convolution", ""): None,
}


class Schedule(NamedTuple):
    name: str
    # The compute capability the export GPU needs; 0: every NVIDIA GPU.
    min_capability: int


def conv_allows_tf32() -> bool:
    """Whether FP32 convolutions may compute in TF32: the setting Inductor's
    Triton conv templates follow, so a convolution keeps its precision whether
    it is routed or not."""
    from torch._dynamo.device_interface import get_interface_for_device

    interface = get_interface_for_device("cuda")
    # Older PyTorch releases read the cuDNN-wide setting directly.
    if not hasattr(interface, "conv_allow_tf32"):
        return torch.backends.cudnn.fp32_precision == "tf32"
    return bool(interface.conv_allow_tf32())


class ConvBindingError(ValueError):
    """The arguments of a conv call cannot be bound to the canonical ABI."""


@dataclass(frozen=True)
class ConvParams:
    stride: tuple[int, ...]
    padding_before: tuple[int, ...]
    padding_after: tuple[int, ...]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int

    @property
    def rank(self) -> int:
        return len(self.stride)


@dataclass(frozen=True)
class ConvCall:
    input: torch.Tensor
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    params: ConvParams

    def signature(self) -> str:
        def describe(t: Optional[torch.Tensor]) -> str:
            if t is None:
                return "None"
            dtype = str(t.dtype).removeprefix("torch.")
            return f"{dtype}{list(t.shape)}@{list(t.stride())}"

        p = self.params
        return (
            f"convolution(input={describe(self.input)}, "
            f"weight={describe(self.weight)}, bias={describe(self.bias)}, "
            f"stride={list(p.stride)}, padding_before={list(p.padding_before)}, "
            f"padding_after={list(p.padding_after)}, dilation={list(p.dilation)}, "
            f"transposed={p.transposed}, output_padding={list(p.output_padding)}, "
            f"groups={p.groups})"
        )


def check_params(p: ConvParams) -> None:
    """Raises ``ConvBindingError`` for canonical arguments outside the ABI."""
    fields = (p.padding_before, p.padding_after, p.dilation, p.output_padding)
    if any(len(f) != p.rank for f in fields) or p.rank not in (1, 2, 3):
        raise ConvBindingError(f"spatial arguments of different lengths: {p}")
    if min(p.stride) <= 0 or min(p.dilation) <= 0 or p.groups <= 0:
        raise ConvBindingError(f"non-positive stride, dilation or groups: {p}")
    if min(p.padding_before) < 0 or min(p.padding_after) < 0:
        raise ConvBindingError(f"negative padding: {p}")
    for o, s, d in zip(p.output_padding, p.stride, p.dilation):
        limit = max(s, d) if p.transposed else 1
        if not 0 <= o < limit:
            raise ConvBindingError(
                f"output_padding {list(p.output_padding)} must be 0 for a forward "
                "convolution and in [0, max(stride, dilation)) for a transposed one"
            )


def _expand(value: Any, rank: int, name: str) -> tuple[int, ...]:
    values = [value] if isinstance(value, int) else list(value)
    if len(values) == 1:
        values = values * rank
    if len(values) != rank:
        raise ConvBindingError(f"{name} has {len(values)} entries for {rank} dims")
    if not all(type(v) is int for v in values):
        raise ConvBindingError(f"{name} is not static: {values}")
    return tuple(values)


def _bind_schema(schema: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> dict:
    names = [a.name for a in schema.arguments]
    if len(args) > len(names) or set(kwargs) - set(names[len(args) :]):
        raise ConvBindingError(f"unexpected arguments for {schema.name}")
    bound = {}
    for i, arg in enumerate(schema.arguments):
        if i < len(args):
            bound[arg.name] = args[i]
        elif arg.name in kwargs:
            bound[arg.name] = kwargs[arg.name]
        elif arg.has_default_value():
            bound[arg.name] = arg.default_value
        else:
            raise ConvBindingError(f"{schema.name} is missing {arg.name}")
    return bound


def _padding(
    padding: Any,
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    kernel: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``padding_before`` and ``padding_after`` for an int list, "valid" or
    "same" (ATen puts the odd element of "same" after)."""
    rank = len(stride)
    if padding == "valid":
        return (0,) * rank, (0,) * rank
    if padding == "same":
        if any(s != 1 for s in stride):
            raise ConvBindingError("padding='same' needs stride 1")
        total = [d * (k - 1) for d, k in zip(dilation, kernel)]
        if not all(type(t) is int for t in total):
            raise ConvBindingError("padding='same' with a symbolic kernel size")
        before = tuple(t // 2 for t in total)
        return before, tuple(t - b for t, b in zip(total, before))
    if isinstance(padding, str):
        raise ConvBindingError(f"unknown padding mode {padding!r}")
    padding = _expand(padding, rank, "padding")
    return padding, padding


def bind_conv(
    target: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    value: Callable[[Any], Any] = lambda a: a,
) -> tuple[Any, Any, Any, ConvParams]:
    """Binds a call of one of ``CONV_OVERLOADS`` to the canonical arguments.

    Returns ``(input, weight, bias, params)``; the first three are the call's
    own arguments (e.g. FX nodes) and ``value`` maps them to tensors.
    """
    schema = target._schema
    key = (schema.name, schema.overload_name)
    if key not in CONV_OVERLOADS:
        raise ConvBindingError(f"{schema.name}.{schema.overload_name} is not a conv")
    bound = _bind_schema(schema, args, kwargs)
    input, weight = value(bound["input"]), value(bound["weight"])
    rank = weight.dim() - 2
    if CONV_OVERLOADS[key] not in (None, rank) or rank not in (1, 2, 3):
        raise ConvBindingError(f"{schema.name} with a {weight.dim()}-D weight")
    if input.dim() != rank + 2:
        raise ConvBindingError(f"unbatched {input.dim()}-D input")
    stride = _expand(bound["stride"], rank, "stride")
    dilation = _expand(bound["dilation"], rank, "dilation")
    transposed = schema.name.startswith("aten::conv_transpose") or bool(
        bound.get("transposed", False)
    )
    # A forward convolution ignores output_padding.
    output_padding = (
        _expand(bound["output_padding"], rank, "output_padding")
        if transposed
        else (0,) * rank
    )
    before, after = _padding(bound["padding"], stride, dilation, weight.shape[2:])
    groups = bound["groups"]
    if type(groups) is not int:
        raise ConvBindingError(f"groups is not static: {groups}")
    params = ConvParams(
        stride, before, after, dilation, transposed, output_padding, groups
    )
    check_params(params)
    return bound["input"], bound["weight"], bound["bias"], params


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides, running = [], 1
    for size in reversed(shape):
        strides.append(running)
        running *= max(size, 1)
    return tuple(reversed(strides))


def _channels_last_order(ndim: int) -> list[int]:
    return [1, *range(ndim - 1, 1, -1), 0]


def channels_last_like(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """``Tensor.suggest_memory_format()`` is channels_last(_3d) for these strides."""
    if len(shape) not in (4, 5) or strides[1] == 0:
        return False
    smallest = 0
    for d in _channels_last_order(len(shape)):
        if shape[d] == 0 or strides[d] < smallest:
            return False
        # ATen falls back to contiguous for ambiguous strides such as N111.
        if d == 0 and smallest == strides[1]:
            return False
        smallest = strides[d] * shape[d]
    return True


def _channels_last_strides(shape: Sequence[int]) -> tuple[int, ...]:
    contiguous = _contiguous_strides(shape)
    # Tensor.to(memory_format=...) keeps strides that already suggest it.
    if channels_last_like(shape, contiguous):
        return contiguous
    strides, running = [0] * len(shape), 1
    for d in _channels_last_order(len(shape)):
        strides[d] = running
        running *= shape[d]
    return tuple(strides)


def _channels_last_output(call: ConvCall) -> bool:
    """Whether ATen's CUDA convolution returns a channels-last output.

    Mirrors the backend choice of a CUDA build with cuDNN: cuDNN picks
    channels-last when the input or the weight suggests it; a transposed
    convolution with ``output_padding >= stride`` and an empty input take
    backends that always return contiguous outputs. A 1-D convolution runs as a
    2-D one, with the input made contiguous and both unsqueezed at dim 2.
    """
    x, w, p = call.input, call.weight, call.params
    if x.shape[0] == 0 or x.shape[1] == 0:
        return False
    if p.transposed and any(o >= s for o, s in zip(p.output_padding, p.stride)):
        return False
    x_shape, x_stride = tuple(x.shape), tuple(x.stride())
    w_shape, w_stride = tuple(w.shape), tuple(w.stride())
    if w.dim() == 3:
        if not _is_contiguous(x_shape, x_stride):
            x_stride = _contiguous_strides(x_shape)
        x_shape, x_stride = _unsqueeze2(x_shape, x_stride)
        w_shape, w_stride = _unsqueeze2(w_shape, w_stride)
    return channels_last_like(x_shape, x_stride) or channels_last_like(
        w_shape, w_stride
    )


def _is_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size != 1:
            if stride != expected:
                return False
            expected *= size
    return True


def _unsqueeze2(
    shape: tuple[int, ...], strides: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (*shape[:2], 1, shape[2]), (*strides[:2], strides[2] * shape[2], strides[2])


def out_shape_and_stride(call: ConvCall) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The size and stride of the output, as ATen's CUDA convolution lays it out."""
    x, w, p = call.input, call.weight, call.params
    kernel = w.shape[2:]
    if p.transposed:
        channels = w.shape[1] * p.groups
        spatial = [
            (i - 1) * s - b - a + d * (k - 1) + o + 1
            for i, k, s, b, a, d, o in zip(
                x.shape[2:],
                kernel,
                p.stride,
                p.padding_before,
                p.padding_after,
                p.dilation,
                p.output_padding,
            )
        ]
    else:
        channels = w.shape[0]
        spatial = [
            (i + b + a - d * (k - 1) - 1) // s + 1
            for i, k, s, b, a, d in zip(
                x.shape[2:],
                kernel,
                p.stride,
                p.padding_before,
                p.padding_after,
                p.dilation,
            )
        ]
    shape = (x.shape[0], channels, *spatial)
    if not _channels_last_output(call):
        return shape, _contiguous_strides(shape)
    if len(shape) == 3:
        strides = _channels_last_strides((shape[0], shape[1], 1, shape[2]))
        return shape, (strides[0], strides[1], strides[3])
    return shape, _channels_last_strides(shape)


def max_offset(t: torch.Tensor) -> int:
    """The largest element offset ``t`` addresses, relative to its data pointer."""
    return sum((n - 1) * s for n, s in zip(t.shape, t.stride()) if n > 0)
