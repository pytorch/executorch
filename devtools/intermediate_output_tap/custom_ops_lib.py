# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Custom op registration for the intermediate-output tap mechanism.

The op `executorch_devtools::tap.Tensor(Tensor x, str reducer_name, int debug_handle) -> Tensor`
is a placeholder whose sole job is to be an unknown-to-every-partitioner FX
node that "uses" a tapped tensor `x`. Because `x` now has a user outside any
partition, every ExecuTorch partitioner must surface `x` as a partition output
(this is the canonical contract enforced in
`executorch/exir/lowered_backend_module.py`).

After `to_edge_transform_and_lower(...)` the tap.Tensor node still exists in
the parent graph; `strip_taps_` (see `_strip_pass.py`) replaces it with either
an identity edge (FULL_TENSOR) or a small reducer subgraph of portable aten
ops.

The op's eager impl computes the named reducer's eager equivalent (e.g.
`min/max/mean/abs_max/...` for STATS). Two reasons for this:

1. **Re-trace safety.** `to_edge_transform_and_lower` re-traces the graph.
   If `tap.Tensor` simply returned `x` literally, the re-traced FX graph
   would treat the tap output as identical to `x` and re-route downstream
   consumers (which would otherwise be reading `x`) through the tap node,
   pulling it into a delegate's input list. Returning a *different* tensor
   (different shape for non-FULL_TENSOR; `x.detach()` for FULL_TENSOR)
   keeps consumers wired to `x` directly so the tap stays a host-only stub
   with the FX `output` node as its sole consumer.
2. **AOT/runtime parity.** Calling `ep_t.module()(*inputs)` then returns the
   same reduced values the runtime emits post-strip, removing the need for
   callers to reapply the reducer themselves.

The dispatch key MUST be `CompositeExplicitAutograd` (not
`CompositeImplicitAutograd`) so the op survives tracing/decomposition;
otherwise it would inline at export time and disappear before partitioning.
This mirrors the pattern in
`executorch/examples/apple/coreml/llama/export_static_llm_coreml.py`.

`reducer_name` and `debug_handle` are stored as op arguments (not just
node.meta) so they survive any meta-stripping pass between `to_edge` and
`strip_taps_`.
"""

from __future__ import annotations

from torch.library import impl, Library

# Library namespace verified collision-free across fbsource as of Nov 2025.
lib: Library = Library("executorch_devtools", "DEF")

lib.define("tap.Tensor(Tensor x, str reducer_name, int debug_handle) -> Tensor")


@impl(lib, "tap.Tensor", "CompositeExplicitAutograd")
def tap_tensor_impl(x, reducer_name, debug_handle):  # noqa: ARG001
    # Defer the import to break a module-import cycle (`_reducers` → torch →
    # custom_ops_lib registration).
    from executorch.devtools.intermediate_output_tap._reducers import get_reducer

    return get_reducer(reducer_name).eager(x)
