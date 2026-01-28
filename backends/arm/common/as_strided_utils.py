# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Utility helpers shared across as_strided_copy handling."""

from __future__ import annotations

import numbers

from collections.abc import Sequence
from typing import Optional, Tuple, TypeVar

import torch
import torch.fx as fx

T = TypeVar("T", bound=Sequence)


def to_int(value: object) -> Optional[int]:
    """Return an int for supported numeric types, otherwise None."""
    if isinstance(value, (numbers.Integral, torch.SymInt)):
        return int(value)
    return None


def maybe_static_sequence(value: object) -> Optional[Sequence]:
    """
    Return a Python sequence for literal or FX-constant values.

    FX exporters often wrap constant lists in nodes where the materialised
    value is stored in ``node.meta["val"]``. This helper unwraps that so the
    rest of the logic can treat them uniformly.
    """
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, fx.Node):
        const_val = value.meta.get("val")
        if isinstance(const_val, Sequence):
            return const_val
        return None
    if isinstance(value, Sequence):
        return value
    return None


def to_int_tuple(value: object) -> Optional[Tuple[int, ...]]:
    """Best-effort conversion of a sequence of integers/SymInts to a tuple[int, ...]."""
    seq = maybe_static_sequence(value)
    if seq is None:
        return None

    result: list[int] = []
    for item in seq:
        converted = to_int(item)
        if converted is None:
            return None
        result.append(converted)
    return tuple(result)


def contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    """Compute row-major contiguous strides for the provided shape."""
    strides = [0] * len(shape)
    running = 1
    for idx in reversed(range(len(shape))):
        dim_val = shape[idx]
        strides[idx] = running if dim_val != 0 else 1
        running *= max(dim_val, 1)
    return tuple(strides)
