# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TapSpec records one tap inserted by `tap_intermediate_outputs(...)`.

A list of TapSpecs is returned to the user from the AOT pass; the user passes
that same list to `Inspector.calculate_numeric_gap_from_taps(...)` at runtime
to demux the flat output tuple back into per-op intermediate values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TapSpec:
    """
    Metadata about a single tap.

    Attributes:
        node_name: The FX node name of the *source* node (the tapped op) at the
            time the AOT pass ran. Useful for debugging / pretty-printing.
        op_target: `str(node.target)` of the source node, e.g.
            "aten.linear.default".
        debug_handle: `node.meta["debug_handle"]` of the source node, or None
            if the source had no debug handle. Set at AOT-pass time. NOT used
            by the Inspector integration directly — the serializer regenerates
            debug_handles, so Inspector aligns by `reducer_node_name` instead.
        output_index: 0-based index into the runtime program's flat output
            tuple where this tap's value lands. Computed at AOT time and stable
            through `to_edge` / `to_backend` / `to_executorch` because we only
            ever *append* to the output node and `OutputSpec`.
        reducer_name: Name of the StatReducer used (e.g. "DEFAULT_STATS").
        fields: Names of the per-element fields in the reducer's output tensor
            (e.g. ("min", "max", "abs_max")). Empty tuple for FULL_TENSOR.
        stack_trace: `node.meta["stack_trace"]` of the source node if present,
            for human-readable error messages.
        reducer_node_name: The FX node name of the post-strip reducer terminal
            node — i.e., the node whose value is surfaced as the runtime tap
            output. Populated by `strip_taps_` when `tap_specs` is passed.
            FX node names survive ETRecord serialization roundtrip, so this
            is the stable bridge `Inspector.calculate_numeric_gap_from_taps`
            uses to find the post-roundtrip handle for alignment.
    """

    node_name: str
    op_target: str
    debug_handle: int | None
    output_index: int
    reducer_name: str
    fields: tuple[str, ...]
    stack_trace: str | None = None
    reducer_node_name: str | None = None
    module_path: str | None = None
