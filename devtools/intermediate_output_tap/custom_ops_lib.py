# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Custom op registration for the intermediate-output tap mechanism.

The op `executorch_devtools::tap.Tensor(Tensor x, str reducer_name, int debug_handle) -> Tensor`
is an identity op whose sole job is to be an unknown-to-every-partitioner FX node
that "uses" a tapped tensor `x`. Because `x` now has a user outside any partition,
every ExecuTorch partitioner must surface `x` as a partition output (this is the
canonical contract enforced in `executorch/exir/lowered_backend_module.py`).

After `to_edge_transform_and_lower(...)` the tap.Tensor node still exists in the
parent graph; `strip_taps_` (see `_strip_pass.py`) replaces it with either an
identity edge (FULL_TENSOR) or a small reducer subgraph of portable aten ops.

The dispatch key MUST be `CompositeExplicitAutograd` (not `CompositeImplicitAutograd`)
so the op survives tracing/decomposition; otherwise it would inline at export time
and disappear before partitioning. This mirrors the pattern in
`executorch/examples/apple/coreml/llama/export_static_llm_coreml.py`.

`reducer_name` and `debug_handle` are stored as op arguments (not just node.meta)
so they survive any meta-stripping pass between `to_edge` and `strip_taps_`.
"""

from __future__ import annotations

from torch.library import impl, Library

# Library namespace verified collision-free across fbsource as of Nov 2025.
lib: Library = Library("executorch_devtools", "DEF")

lib.define("tap.Tensor(Tensor x, str reducer_name, int debug_handle) -> Tensor")


@impl(lib, "tap.Tensor", "CompositeExplicitAutograd")
def tap_tensor_impl(x, reducer_name, debug_handle):  # noqa: ARG001
    return x
