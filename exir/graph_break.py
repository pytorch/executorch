# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for inserting graph breaks to force backend partitioning boundaries.

A graph break is an identity custom op that is not recognized by any backend's
``CapabilityBasedPartitioner``, which prevents the partitioner from merging
nodes across the break into a single partition. After partitioning, the graph
break ops should be removed via :func:`remove_graph_break_ops`.

Typical workflow::

    from executorch.exir.graph_break import BlockWithGraphBreak, remove_graph_break_ops

    # 1. Wrap model blocks with graph breaks before export
    model.layers[0] = BlockWithGraphBreak(model.layers[0], break_before=True)
    model.layers[-1] = BlockWithGraphBreak(model.layers[-1], break_before=False)

    # 2. Export and lower (partitioner creates splits at graph breaks)
    edge_manager = to_edge_transform_and_lower(exported_programs, ...)

    # 3. Remove graph break ops from the edge program
    remove_graph_break_ops(edge_manager)

    # 4. Convert to ExecuTorch
    et_program = edge_manager.to_executorch()
"""

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.library import impl, Library

__all__ = [
    "GraphBreakModule",
    "BlockWithGraphBreak",
    "remove_graph_break_ops",
]

# Register the custom op. The op is an identity function that is unknown to
# backends, forcing the partitioner to create a boundary at this node.
_lib = Library("executorch_utils", "DEF")
_lib.define("graph_break.Tensor(Tensor x) -> Tensor")


@impl(_lib, "graph_break.Tensor", "CompositeExplicitAutograd")
def _graph_break_impl(x: torch.Tensor) -> torch.Tensor:
    return x


class GraphBreakModule(nn.Module):
    """Module that applies a graph break to every tensor in ``*args``.

    Non-tensor arguments are passed through unchanged.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[Any, ...]:
        return tuple(
            (
                torch.ops.executorch_utils.graph_break.Tensor(a)
                if isinstance(a, torch.Tensor)
                else a
            )
            for a in args
        )


class BlockWithGraphBreak(nn.Module):
    """Wraps a module with a graph break inserted before or after it.

    Args:
        block: The module to wrap.
        break_before: If ``True``, the graph break is inserted before ``block``.
            If ``False``, it is inserted after.
    """

    def __init__(self, block: nn.Module, break_before: bool = True) -> None:
        super().__init__()
        self.graph_break = GraphBreakModule()
        self.block = block
        self.break_before = break_before

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.break_before:
            new_args = self.graph_break(*args)
            return self.block(*new_args, **kwargs)
        else:
            out = self.block(*args, **kwargs)
            out = self.graph_break(*out)
            return out


def remove_graph_break_ops(edge_manager: Any) -> None:
    """Remove all graph break ops from an edge manager's methods.

    This should be called after ``to_edge_transform_and_lower`` (which creates
    partition boundaries at graph breaks) but before ``to_executorch()``.

    Args:
        edge_manager: The ``EdgeProgramManager`` to modify in-place.
    """
    from executorch.exir.dialects._ops import ops as exir_ops

    for method_name in edge_manager.methods:
        ep = edge_manager.exported_program(method_name)
        for n in ep.graph_module.graph.nodes:
            if n.target == exir_ops.edge.executorch_utils.graph_break.Tensor:
                n.replace_all_uses_with(n.args[0])
        ep.graph_module.graph.eliminate_dead_code()
