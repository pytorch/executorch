# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which dense matrix multiplications become ``triton::linear``.

``plan(node)`` is the replacement pass's preflight (see ``triton/routing.py``)
for the forms the CUDA backend receives ``x @ weight.T + bias`` in:

1. ``linear(x, weight, bias=None)``;
2. ``matmul(x, t(weight))`` with a 2-D right operand;
3. ``mm(x, t(weight))``;
4. ``addmm(bias, x, t(weight), beta=1, alpha=1)``;

where ``t`` is the 2-D transpose ``t_copy``, ``permute_copy(dims=[1, 0])`` or
``transpose_copy`` of the dims 0 and 1, ``weight`` an uncomputed graph input
(a placeholder, such as a parameter) holding a contiguous ``[N, K]`` tensor and
``bias`` None or a contiguous ``[N]`` tensor. Every tensor is strict FP32
(with TF32 matmuls disallowed) on one CUDA device. The replacement takes the
placeholder itself; only the GEMM node is replaced, and a transpose left
without users is removed with it.

The regular lowering (Inductor) runs every such call. ``triton::linear`` runs
the calls its layout ABI accepts (``linear.layout_reason``). The family opts
in to measured routing; its static choice, which decides with measurement off
or for a call that cannot be measured, is ``triton::linear`` for the
matrix-vector class ``worth_routing`` takes and the regular lowering
otherwise.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
from executorch.backends.cuda.triton.kernels.gemm.gemv import MAX_ROWS
from executorch.backends.cuda.triton.kernels.gemm.linear import (
    describe,
    layout_reason,
    matmul_allows_tf32,
    output_shape,
    row_stride,
    signature,
)
from executorch.backends.cuda.triton.routing import (
    Candidate,
    is_static,
    REGULAR_LOWERING,
    regular_only,
    ReplacementPlan,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torch.fx.operator_schemas import normalize_function

logger: logging.Logger = logging.getLogger(__name__)

KIND = "gemm"
REPLACEMENT = "linear"

GEMM_TARGETS = (
    exir_ops.edge.aten.linear.default,
    exir_ops.edge.aten.matmul.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.addmm.default,
)
TRANSPOSES = (
    exir_ops.edge.aten.t_copy.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.transpose_copy.int,
)


class _Refused(Exception):
    pass


# The matrix-vector class the static choice routes (``gemv_boundary_calls``
# lists its corners; ``benchmarks/gemm/benchmark_linear.py`` measures them).
GEMV_MIN_N = 64
GEMV_MIN_K = 1024
# Elements per 16 bytes: the rows of both operands and their bases must be
# aligned to it for the streamed slabs to load as vectors.
GEMV_K_ALIGNMENT = 4
# The largest weights measured.
GEMV_MAX_N = 262144
GEMV_MAX_K = 16384
GEMV_MAX_WEIGHT = 2**29


def worth_routing(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[str]:
    """Why a call ``triton::linear`` runs is left to the regular lowering when
    it is not measured, or None: only calls of at most ``MAX_ROWS`` rows
    whose weight is large enough to keep the matrix-vector schedule's memory
    stream busy, with contiguous input rows, and with the rows and bases of
    both operands aligned for vector loads, are routed."""
    m = math.prod(x.shape[:-1])
    n, k = weight.shape
    if m > MAX_ROWS:
        return f"{m} rows: only calls of at most {MAX_ROWS} rows are routed"
    if x.stride(-1) != 1:
        return f"input strides {list(x.stride())}: its rows are not contiguous"
    if k % GEMV_K_ALIGNMENT:
        return (
            f"K={k} is not a multiple of {GEMV_K_ALIGNMENT}: the weight's rows are "
            "not aligned for vector loads"
        )
    rows = row_stride(x.shape, x.stride())
    if m > 1 and rows % GEMV_K_ALIGNMENT:
        return (
            f"input row stride {rows} is not a multiple of {GEMV_K_ALIGNMENT}: its "
            "rows are not aligned for vector loads"
        )
    for name, t in (("input", x), ("weight", weight)):
        if t.storage_offset() % GEMV_K_ALIGNMENT:
            return (
                f"{name} storage offset {t.storage_offset()} is not a multiple of "
                f"{GEMV_K_ALIGNMENT}: its base is not aligned for vector loads"
            )
    if not (
        GEMV_MIN_N <= n <= GEMV_MAX_N
        and GEMV_MIN_K <= k <= GEMV_MAX_K
        and n * k <= GEMV_MAX_WEIGHT
    ):
        return (
            f"a [{n}, {k}] weight is outside the matrix-vector class (N in "
            f"[{GEMV_MIN_N}, {GEMV_MAX_N}], K in [{GEMV_MIN_K}, {GEMV_MAX_K}], "
            f"at most {GEMV_MAX_WEIGHT} elements)"
        )
    return None


def gemv_boundary_calls() -> list[tuple[int, int, int]]:
    """``(rows, N, K)`` at every corner of the matrix-vector class, with the
    first weight sizes past its floors that leave tile remainders."""
    return [
        (m, n, k)
        for m in (1, 2, MAX_ROWS)
        for k in (GEMV_MIN_K, GEMV_MIN_K + GEMV_K_ALIGNMENT, GEMV_MAX_K)
        for n in (GEMV_MIN_N, GEMV_MIN_N + 1, min(GEMV_MAX_N, GEMV_MAX_WEIGHT // k))
    ]


def _bind(node: Node) -> dict[str, Any]:
    """The node's arguments by name, defaults included."""
    bound = normalize_function(
        node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )
    if bound is None:
        raise _Refused(f"cannot bind the arguments of {node.target}")
    return bound.kwargs


def _value(arg: Any) -> Any:
    return arg.meta.get("val") if isinstance(arg, Node) else arg


def _transposed(node: Any) -> Node:
    """The ``[N, K]`` node whose 2-D transpose ``node`` is."""
    swapped = False
    if (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target in TRANSPOSES
    ):
        if node.target == exir_ops.edge.aten.t_copy.default:
            swapped = True
        elif node.target == exir_ops.edge.aten.permute_copy.default:
            swapped = [d % 2 for d in _bind(node)["dims"]] == [1, 0]
        elif node.target == exir_ops.edge.aten.transpose_copy.int:
            bound = _bind(node)
            swapped = bound["dim0"] % 2 != bound["dim1"] % 2
    source = node.args[0] if swapped else None
    value = _value(source)
    if not isinstance(value, torch.Tensor) or value.dim() != 2:
        raise _Refused("the right operand is not a transposed [N, K] weight")
    return source


def _forms(node: Node) -> tuple[Any, Any, Any]:
    """``(x, weight, bias)`` of a GEMM node, as its arguments (nodes)."""
    bound = _bind(node)
    if node.target == exir_ops.edge.aten.linear.default:
        return bound["input"], bound["weight"], bound["bias"]
    if node.target == exir_ops.edge.aten.addmm.default:
        if bound["beta"] != 1 or bound["alpha"] != 1:
            raise _Refused(f"beta={bound['beta']}, alpha={bound['alpha']}")
        bias = _value(bound["input"])
        if not isinstance(bias, torch.Tensor) or bias.dim() != 1:
            shown = describe(bias) if isinstance(bias, torch.Tensor) else repr(bias)
            raise _Refused(f"the addend {shown} is not a [N] bias")
        return bound["mat1"], _transposed(bound["mat2"]), bound["input"]
    other = bound["mat2"] if "mat2" in bound else bound["other"]
    other_value = _value(other)
    if isinstance(other_value, torch.Tensor) and other_value.dim() != 2:
        raise _Refused(f"a matmul with a {other_value.dim()}-D right operand")
    return bound["input"], _transposed(other), None


def _check(node: Node, x: Any, weight: Any, bias: Any) -> None:
    if not isinstance(weight, Node) or weight.op != "placeholder":
        producer = weight.name if isinstance(weight, Node) else repr(weight)
        raise _Refused(f"the weight is computed in the graph (by {producer})")
    xv, wv, bv = (_value(a) for a in (x, weight, bias))
    expected = node.meta.get("val")
    if not all(
        isinstance(t, torch.Tensor) and is_static(t) for t in (xv, wv, expected)
    ) or not is_static(bv):
        raise _Refused("symbolic shapes or missing fake tensors")
    if xv.dtype != torch.float32:
        raise _Refused(f"dtype {xv.dtype}: only strict FP32 GEMMs are routed")
    if matmul_allows_tf32():
        raise _Refused("TF32 matmuls are allowed: only strict FP32 GEMMs are routed")
    if not wv.is_contiguous():
        raise _Refused(
            f"weight strides {list(wv.stride())} are not a contiguous [N, K]"
        )
    if expected.dtype != torch.float32 or tuple(expected.shape) != output_shape(xv, wv):
        raise _Refused(
            f"output {describe(expected)} is not the float32 "
            f"{list(output_shape(xv, wv))} of the operands"
        )
    reason = layout_reason(xv, wv, bv, expected.stride())
    if reason is not None:
        raise _Refused(reason)


def plan(node: Node) -> ReplacementPlan:
    """The implementations of a GEMM node; never changes the graph."""
    if torch.version.hip is not None:
        return regular_only(KIND, "ROCm")
    try:
        x, weight, bias = _forms(node)
        _check(node, x, weight, bias)
    except _Refused as e:
        return regular_only(KIND, str(e))
    except Exception as e:  # noqa: BLE001 - the regular lowering runs it
        logger.warning("GEMM preflight failed for %s: %r", node.name, e)
        return regular_only(KIND, f"preflight failed: {e!r}")
    xv, wv, bv = (_value(a) for a in (x, weight, bias))
    strides = list(node.meta["val"].stride())
    replacement = Candidate(
        REPLACEMENT,
        None,
        "strict FP32 linear",
        torch.ops.triton.linear.default,
        (x, weight, bias, strides),
    )
    worth = worth_routing(xv, wv, bv)
    return ReplacementPlan(
        KIND,
        (Candidate(REGULAR_LOWERING), replacement),
        REGULAR_LOWERING if worth is not None else REPLACEMENT,
        worth or "",
        signature(xv, wv, bv, strides),
        measure=True,
    )
