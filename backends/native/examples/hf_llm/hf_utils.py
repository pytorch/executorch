# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Report native lowering for exported Hugging Face models."""

import collections
import inspect
import operator

import torch
from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_submodules,
)
from torch.export import ExportedProgram
from transformers.cache_utils import StaticLayer


class PositionedStaticLayer(StaticLayer):
    """A static cache layer that writes K/V, cast to its own dtype, at ``positions``.

    transformers 5 writes at ``arange(n) + cumulative_length``. The native runtime
    only accepts row indices that are a graph input or an arange, so the positions
    are the ``cache_position`` input of the graph instead.
    """

    positions: torch.Tensor

    @classmethod
    def replacing(cls, old: StaticLayer, dtype: torch.dtype) -> "PositionedStaticLayer":
        layer = cls(old.max_cache_len)
        keys = old.keys[:, :, :1].to(dtype)
        values = old.values[:, :, :1].to(dtype)
        # transformers 4 sizes both buffers from the keys alone.
        if len(inspect.signature(layer.lazy_initialization).parameters) == 1:
            layer.lazy_initialization(keys)
        else:
            layer.lazy_initialization(keys, values)
        return layer

    def update(self, key_states, value_states, *args, **kwargs):
        self.keys.index_copy_(2, self.positions, key_states.to(self.dtype))
        self.values.index_copy_(2, self.positions, value_states.to(self.dtype))
        return self.keys, self.values


def histogram(gm: torch.fx.GraphModule) -> collections.Counter:
    """Count call_function targets, ignoring delegate plumbing."""
    return collections.Counter(
        str(n.target)
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target not in (executorch_call_delegate, operator.getitem)
    )


def report_lowering(ep: ExportedProgram, label: str) -> None:
    """Lower ``ep`` for PTN and print delegated vs. residual op histograms."""
    print(f"== {label}: exported aten ops ==", flush=True)
    for k, v in sorted(histogram(ep.graph_module).items()):
        print(f"  {v:4d} {k}")

    edge = to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[
            NativePartitioner(external_constants_tag=None, _serialize_as_ptn=True)
        ],
        compile_config=get_default_compile_config(),
    )
    gm = edge.exported_program().graph_module
    lowered = get_lowered_submodules(gm)
    print(f"== {label}: {len(lowered)} native delegate(s) ==")
    delegated: collections.Counter = collections.Counter()
    for _, module, _ in lowered:
        delegated += histogram(module.original_module.graph_module)
    for k, v in sorted(delegated.items()):
        print(f"  {v:4d} {k}")
    print(f"== {label}: residual (not delegated) ==")
    residual = histogram(gm)
    if not residual:
        print("  (none)")
    for k, v in sorted(residual.items()):
        print(f"  {v:4d} {k}")
