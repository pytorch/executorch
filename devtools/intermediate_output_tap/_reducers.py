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

The emit functions cast to fp32 first for cross-backend numerical stability
and use full-tensor reductions (no `dim=`) so the result is a stable shape
regardless of the source tensor's rank.

For v1 we ship: FULL_TENSOR, ABS_MAX_ONLY, MIN_MAX_MEAN, DEFAULT_STATS.
HISTOGRAM_64 is deferred (`aten.histc` has restricted edge support).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from executorch.exir.dialects._ops import ops as exir_ops


if TYPE_CHECKING:
    import torch.fx as fx


# --- Reducer dataclass ---------------------------------------------------

EmitFn = Callable[["fx.Graph", "fx.Node"], "fx.Node"]


@dataclass(frozen=True)
class StatReducer:
    """
    A reducer specification. `emit` is invoked by `strip_taps_` to materialise
    the reducer subgraph in the post-lowering graph.

    `name` is what the user types and what's stored on each TapSpec.
    `fields` enumerates the columns of the 1-D output tensor (empty for
    FULL_TENSOR, which preserves the original tensor shape).
    """

    name: str
    fields: tuple[str, ...]
    emit: EmitFn


# --- Helpers -------------------------------------------------------------


def _cast_fp32(graph: "fx.Graph", x: "fx.Node") -> "fx.Node":
    """Insert a fp32 cast (no-op semantically if already fp32)."""
    # exir_ops.edge.dim_order_ops._to_dim_order_copy.default exists for edge dialect,
    # but the simpler aten._to_copy variant is broadly supported.
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


# --- Built-in reducers ---------------------------------------------------


def _emit_full_tensor(_graph: "fx.Graph", src: "fx.Node") -> "fx.Node":
    """Identity — return the source node directly. strip_taps_ will splice."""
    return src


FULL_TENSOR: StatReducer = StatReducer(
    name="FULL_TENSOR",
    fields=(),
    emit=_emit_full_tensor,
)


def _emit_abs_max(graph: "fx.Graph", src: "fx.Node") -> "fx.Node":
    f = _cast_fp32(graph, src)
    abs_x = graph.call_function(exir_ops.edge.aten.abs.default, args=(f,))
    return _scalar_node(graph, exir_ops.edge.aten.amax.default, abs_x)


ABS_MAX_ONLY: StatReducer = StatReducer(
    name="ABS_MAX_ONLY",
    fields=("abs_max",),
    emit=_emit_abs_max,
)


def _emit_min_max_mean(graph: "fx.Graph", src: "fx.Node") -> "fx.Node":
    f = _cast_fp32(graph, src)
    mn = _scalar_node(graph, exir_ops.edge.aten.amin.default, f)
    mx = _scalar_node(graph, exir_ops.edge.aten.amax.default, f)
    me = _scalar_node(graph, exir_ops.edge.aten.mean.default, f)
    return _stack(graph, [mn, mx, me])


MIN_MAX_MEAN: StatReducer = StatReducer(
    name="MIN_MAX_MEAN",
    fields=("min", "max", "mean"),
    emit=_emit_min_max_mean,
)


def _emit_default_stats(graph: "fx.Graph", src: "fx.Node") -> "fx.Node":
    """
    Default stats: (min, max, mean, abs_max) — 4 floats.

    NOTE: nan_count/inf_count/std are intentionally excluded because the
    underlying portable kernels (`isnan`, `isinf`, `sum.dtype`, `std.*`)
    don't all have out variants registered in ExecuTorch's default runtime
    op table, which fails memory planning or runtime method-load. If you
    need them, supply a custom StatReducer.
    """
    f = _cast_fp32(graph, src)
    mn = _scalar_node(graph, exir_ops.edge.aten.amin.default, f)
    mx = _scalar_node(graph, exir_ops.edge.aten.amax.default, f)
    me = _scalar_node(graph, exir_ops.edge.aten.mean.default, f)

    abs_x = graph.call_function(exir_ops.edge.aten.abs.default, args=(f,))
    abs_max = _scalar_node(graph, exir_ops.edge.aten.amax.default, abs_x)

    return _stack(graph, [mn, mx, me, abs_max])


DEFAULT_STATS: StatReducer = StatReducer(
    name="DEFAULT_STATS",
    fields=("min", "max", "mean", "abs_max"),
    emit=_emit_default_stats,
)


# --- Registry -------------------------------------------------------------

_BUILTIN_REDUCERS: dict[str, StatReducer] = {
    r.name: r
    for r in (FULL_TENSOR, ABS_MAX_ONLY, MIN_MAX_MEAN, DEFAULT_STATS)
}


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
