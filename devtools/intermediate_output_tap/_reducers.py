# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Stat reducers used by `tap_intermediate_outputs`.

A `StatReducer` is a small specification consumed by `strip_taps_` (after
`to_backend`) to materialise a portable reducer subgraph in place of the
`executorch_devtools::tap.Tensor` placeholder.

`emit(graph, src_node) -> fx.Node` builds the reducer subgraph in `graph`
just before the placeholder, using the source tensor `src_node` as input,
and returns the final node whose output replaces the placeholder's output.

`eager(tensor) -> tensor` is the pure-torch equivalent that callers can use
to reproduce, in eager mode, what the runtime will compute. `tap.Tensor`'s
own dispatch impl uses this to produce the *reduced* value at AOT time, so
that `ep.module()(*inputs)` returns the same flat outputs as the runtime.

The emit functions cast to fp32 first for cross-backend numerical stability
and produce a fixed-shape output (0-D or 1-D) regardless of the source
tensor's rank, so callers don't need to track per-tap shapes.

We ship two built-ins:

* `FULL_TENSOR` — identity. The whole source tensor is surfaced.
* `STATS` — a comprehensive bundle of debugging-friendly scalars:
  min, max, mean, abs_max, abs_mean, std, rms, l1_norm, l2_norm,
  nan_count, inf_count, zero_count, p99_abs.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from executorch.exir.dialects._ops import ops as exir_ops


if TYPE_CHECKING:
    import torch.fx as fx


# --- Reducer dataclass ---------------------------------------------------

EmitFn = Callable[["fx.Graph", "fx.Node"], "fx.Node"]
EagerFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class StatReducer:
    """
    A reducer specification.

    `emit` is invoked by `strip_taps_` to materialise the reducer subgraph
    in the post-lowering FX graph. `eager` is the equivalent pure-torch
    implementation, used by callers that want to reproduce what the runtime
    will compute (e.g. AOT-vs-runtime comparisons without a debugger).

    `name` is what the user types and what's stored on each TapSpec.
    `fields` enumerates the columns of the 1-D output tensor (empty for
    FULL_TENSOR which preserves a tensor of values).
    """

    name: str
    fields: tuple[str, ...]
    emit: EmitFn
    eager: EagerFn


# --- Helpers -------------------------------------------------------------


def _cast_fp32(graph: "fx.Graph", x: "fx.Node") -> "fx.Node":
    """Insert a fp32 cast (no-op semantically if already fp32)."""
    return graph.call_function(
        exir_ops.edge.aten._to_copy.default,
        args=(x,),
        kwargs={"dtype": torch.float32},
    )


def _scalar_node(graph: "fx.Graph", op, x: "fx.Node") -> "fx.Node":
    """Call a full-reduction op (amin/amax/mean/sum) producing a 0-d tensor."""
    return graph.call_function(op, args=(x,))


def _stack(graph: "fx.Graph", scalars: list["fx.Node"]) -> "fx.Node":
    """Stack a list of 0-d tensors into a 1-D tensor."""
    return graph.call_function(
        exir_ops.edge.aten.stack.default,
        args=(scalars,),
        kwargs={"dim": 0},
    )


def _abs(graph: "fx.Graph", x: "fx.Node") -> "fx.Node":
    return graph.call_function(exir_ops.edge.aten.abs.default, args=(x,))


def _square(graph: "fx.Graph", x: "fx.Node") -> "fx.Node":
    return graph.call_function(exir_ops.edge.aten.pow.Tensor_Scalar, args=(x, 2.0))


def _sqrt(graph: "fx.Graph", x: "fx.Node") -> "fx.Node":
    return graph.call_function(exir_ops.edge.aten.sqrt.default, args=(x,))


def _full_sum(graph: "fx.Graph", x: "fx.Node") -> "fx.Node":
    """Full-tensor sum via aten.sum.dim_IntList(dim=[]) — portable + has out variant."""
    return graph.call_function(exir_ops.edge.aten.sum.dim_IntList, args=(x, []))


def _bool_to_fp32_count(graph: "fx.Graph", mask: "fx.Node") -> "fx.Node":
    """Sum of a bool mask cast to fp32 → a 0-d fp32 count."""
    casted = graph.call_function(
        exir_ops.edge.aten._to_copy.default,
        args=(mask,),
        kwargs={"dtype": torch.float32},
    )
    return _full_sum(graph, casted)


# --- FULL_TENSOR ---------------------------------------------------------


def _emit_full_tensor(_graph: "fx.Graph", src: "fx.Node") -> "fx.Node":
    return src


def _eager_full_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


FULL_TENSOR: StatReducer = StatReducer(
    name="FULL_TENSOR",
    fields=(),
    emit=_emit_full_tensor,
    eager=_eager_full_tensor,
)


# --- STATS ---------------------------------------------------------------


_STATS_FIELDS: tuple[str, ...] = (
    "min",
    "max",
    "mean",
    "abs_max",
    "abs_mean",
    "std",
    "rms",
    "l1_norm",
    "l2_norm",
    "nan_count",
    "inf_count",
    "zero_count",
    "p99_abs",
)


def _emit_stats(graph: "fx.Graph", src: "fx.Node") -> "fx.Node":
    f = _cast_fp32(graph, src)
    abs_f = _abs(graph, f)
    sq_f = _square(graph, f)

    mn = _scalar_node(graph, exir_ops.edge.aten.amin.default, f)
    mx = _scalar_node(graph, exir_ops.edge.aten.amax.default, f)
    me = _scalar_node(graph, exir_ops.edge.aten.mean.default, f)
    abs_max = _scalar_node(graph, exir_ops.edge.aten.amax.default, abs_f)
    abs_mean = _scalar_node(graph, exir_ops.edge.aten.mean.default, abs_f)

    sum_sq = _full_sum(graph, sq_f)
    mean_sq = _scalar_node(graph, exir_ops.edge.aten.mean.default, sq_f)
    rms = _sqrt(graph, mean_sq)

    # std = sqrt( E[x^2] - E[x]^2 ); avoids aten.var which lacks an out variant.
    me_sq_scalar = graph.call_function(
        exir_ops.edge.aten.pow.Tensor_Scalar, args=(me, 2.0)
    )
    var = graph.call_function(
        exir_ops.edge.aten.sub.Tensor, args=(mean_sq, me_sq_scalar)
    )
    # Variance can be slightly negative due to fp roundoff; clamp at 0 via abs.
    var = graph.call_function(exir_ops.edge.aten.abs.default, args=(var,))
    std = _sqrt(graph, var)

    l1 = _full_sum(graph, abs_f)
    l2 = _sqrt(graph, sum_sq)

    nan_mask = graph.call_function(exir_ops.edge.aten.isnan.default, args=(f,))
    nan_count = _bool_to_fp32_count(graph, nan_mask)

    inf_mask = graph.call_function(exir_ops.edge.aten.isinf.default, args=(f,))
    inf_count = _bool_to_fp32_count(graph, inf_mask)

    zero_mask = graph.call_function(exir_ops.edge.aten.eq.Scalar, args=(f, 0.0))
    zero_count = _bool_to_fp32_count(graph, zero_mask)

    # p99_abs: use topk on flattened |x| to get the k-th largest, where
    # k = max(1, ceil(numel * 0.01)). Numel is read from the source's
    # FakeTensor at graph-build time.
    fake = src.meta.get("val")
    numel = int(fake.numel()) if fake is not None else 1
    k = max(1, (numel + 99) // 100)  # ceil(numel/100)
    abs_flat = graph.call_function(
        exir_ops.edge.aten.view_copy.default, args=(abs_f, [-1])
    )
    topk_out = graph.call_function(
        exir_ops.edge.aten.topk.default,
        args=(abs_flat, k),
        kwargs={"dim": -1, "largest": True, "sorted": True},
    )
    topk_values = graph.call_function(operator.getitem, args=(topk_out, 0))
    p99_abs = graph.call_function(
        exir_ops.edge.aten.select_copy.int, args=(topk_values, 0, k - 1)
    )

    return _stack(
        graph,
        [
            mn,
            mx,
            me,
            abs_max,
            abs_mean,
            std,
            rms,
            l1,
            l2,
            nan_count,
            inf_count,
            zero_count,
            p99_abs,
        ],
    )


def _eager_stats(t: torch.Tensor) -> torch.Tensor:
    f = t.detach().to(torch.float32)
    abs_f = f.abs()
    sq = f.pow(2.0)

    # std via E[x^2] - E[x]^2 (population variance) to match the emit subgraph.
    if f.numel() > 0:
        var = (sq.mean() - f.mean().pow(2)).abs()
        std = var.sqrt()
    else:
        std = torch.tensor(0.0)

    sum_sq = sq.sum()
    rms = sq.mean().sqrt()
    l1 = abs_f.sum()
    l2 = sum_sq.sqrt()

    nan_count = torch.isnan(f).to(torch.float32).sum()
    inf_count = torch.isinf(f).to(torch.float32).sum()
    zero_count = (f == 0).to(torch.float32).sum()

    numel = f.numel()
    k = max(1, (numel + 99) // 100)
    if numel > 0:
        topk_vals = torch.topk(abs_f.reshape(-1), k=k, largest=True, sorted=True).values
        p99_abs = topk_vals[k - 1]
    else:
        p99_abs = torch.tensor(float("nan"))

    return torch.stack(
        [
            f.amin(),
            f.amax(),
            f.mean(),
            abs_f.amax(),
            abs_f.mean(),
            std,
            rms,
            l1,
            l2,
            nan_count,
            inf_count,
            zero_count,
            p99_abs,
        ],
        dim=0,
    )


STATS: StatReducer = StatReducer(
    name="STATS",
    fields=_STATS_FIELDS,
    emit=_emit_stats,
    eager=_eager_stats,
)


# --- Registry -------------------------------------------------------------

_BUILTIN_REDUCERS: dict[str, StatReducer] = {r.name: r for r in (FULL_TENSOR, STATS)}


def get_reducer(name_or_reducer: str | StatReducer) -> StatReducer:
    """Look up a built-in by name, or return a user-supplied StatReducer as-is."""
    if isinstance(name_or_reducer, StatReducer):
        return name_or_reducer
    if name_or_reducer not in _BUILTIN_REDUCERS:
        raise ValueError(
            f"Unknown reducer {name_or_reducer!r}; "
            f"available: {sorted(_BUILTIN_REDUCERS)}"
        )
    return _BUILTIN_REDUCERS[name_or_reducer]
