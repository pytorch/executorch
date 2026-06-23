# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node


# Each rewrite returns the TOSA dialect node spec for one supported ATen
# activation op, preserving args unless TOSA requires normalized attributes.
def rewrite_erf(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    return DialectNodeSpec(
        exir_ops.backend.tosa.ERF.default,
        node.args,
        dict(node.kwargs),
    )


def rewrite_sigmoid(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    return DialectNodeSpec(
        exir_ops.backend.tosa.SIGMOID.default,
        node.args,
        dict(node.kwargs),
    )


def rewrite_tanh(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    return DialectNodeSpec(
        exir_ops.backend.tosa.TANH.default,
        node.args,
        dict(node.kwargs),
    )


def _extract_dtype(node: Node) -> torch.dtype | None:
    value = node.meta.get("val")
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    return getattr(value, "dtype", None)


def _dtype_bounds(dtype: torch.dtype) -> tuple[int | float, int | float]:
    if dtype.is_floating_point:
        fp_info = torch.finfo(dtype)
        return fp_info.min, fp_info.max

    int_info = torch.iinfo(dtype)
    return int_info.min, int_info.max


def _is_tosa_clamp_dtype_supported(dtype: torch.dtype) -> bool:
    tosa_spec = get_context_spec()

    if dtype == torch.int8:
        return tosa_spec.support_integer()

    if dtype == torch.int16:
        return tosa_spec.support_integer() and tosa_spec.support_extension("int16")

    if dtype in (torch.float16, torch.float32):
        return tosa_spec.support_float()

    if dtype == torch.bfloat16:
        return tosa_spec.support_float() and tosa_spec.support_extension("bf16")

    return False


def _normalize_clamp_bound(
    bound,
    *,
    dtype: torch.dtype,
    default: int | float,
) -> int | float | None:
    if bound is None:
        return default
    if isinstance(bound, bool):
        return None
    if dtype.is_floating_point:
        if isinstance(bound, (int, float)):
            return float(bound)
        return None
    if isinstance(bound, int):
        return bound
    return None


def _get_min_max_arguments(
    node: Node, dtype: torch.dtype
) -> tuple[int | float, int | float] | None:
    dtype_min, dtype_max = _dtype_bounds(dtype)
    min_val = _normalize_clamp_bound(
        node.args[1] if len(node.args) > 1 else node.kwargs.get("min"),
        dtype=dtype,
        default=dtype_min,
    )
    max_val = _normalize_clamp_bound(
        node.args[2] if len(node.args) > 2 else node.kwargs.get("max"),
        dtype=dtype,
        default=dtype_max,
    )
    if min_val is None or max_val is None:
        return None
    return min_val, max_val


def rewrite_clamp(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec | None:
    dtype = _extract_dtype(node)
    if dtype is None or not _is_tosa_clamp_dtype_supported(dtype):
        return None

    min_max_args = _get_min_max_arguments(node, dtype)
    if min_max_args is None:
        return None

    return DialectNodeSpec(
        exir_ops.backend.tosa.CLAMP.default,
        (node.args[0], *min_max_args),
    )
