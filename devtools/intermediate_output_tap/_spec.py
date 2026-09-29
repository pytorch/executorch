# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TapSpec records one tap inserted by `tap_intermediate_outputs(...)`.

A list of TapSpecs is returned to the user from the AOT pass; the user uses
the `output_index` on each spec to demux the runtime program's flat output
tuple back into per-op intermediate values (e.g. via
`compare_aot_runtime_dataframe`).
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
        output_index: 0-based index into the runtime program's flat output
            tuple where this tap's value lands. Computed at AOT time and stable
            through `to_edge` / `to_backend` / `to_executorch` because we only
            ever *append* to the output node and `OutputSpec`.
        reducer_name: Name of the StatReducer used (e.g. "STATS").
        fields: Names of the per-element fields in the reducer's output tensor
            (e.g. ("min", "max", "abs_max")). Empty tuple for FULL_TENSOR.
        stack_trace: `node.meta["stack_trace"]` of the source node if present,
            for human-readable error messages.
        module_path: The `nn_module_stack` path of the source node, e.g.
            "layers.1.attention.wvs.0", or None if not available.
        module_class: Bare class name of the leaf nn.Module the source node
            ran inside (e.g. "Linear", "_RMSNorm"), or None if not available.
    """

    node_name: str
    op_target: str
    output_index: int
    reducer_name: str
    fields: tuple[str, ...]
    stack_trace: str | None = None
    module_path: str | None = None
    module_class: str | None = None
