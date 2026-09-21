# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

from .pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)


@register_pattern_detector("swiglu")
def find_swiglu_pattern(node: torch.fx.Node) -> Optional[PatternMatch]:
    if node.target != exir_ops.edge.aten.mul.Tensor:
        return None

    for silu, up in (node.args, node.args[::-1]):
        if (
            not isinstance(silu, torch.fx.Node)
            or not isinstance(up, torch.fx.Node)
            or silu.target != exir_ops.edge.aten.mul.Tensor
            or len(silu.users) != 1
        ):
            continue
        for gate, sigmoid in (silu.args, silu.args[::-1]):
            if (
                not isinstance(gate, torch.fx.Node)
                or not isinstance(sigmoid, torch.fx.Node)
                or sigmoid.target != exir_ops.edge.aten.sigmoid.default
                or sigmoid.args != (gate,)
                or len(sigmoid.users) != 1
            ):
                continue
            values = [n.meta.get("val") for n in (gate, up, sigmoid, silu, node)]
            if not all(isinstance(v, torch.Tensor) for v in values):
                continue
            if not all(v.dtype == values[0].dtype for v in values):
                continue
            if values[0].dtype not in (torch.float16, torch.float32):
                continue
            return PatternMatch(
                [gate, up], [node], [sigmoid, silu, node], anchor_node=node
            )
    return None


@register_pattern_replacement("swiglu")
def replace_swiglu_pattern(
    ep: ExportedProgram, graph_module: torch.fx.GraphModule, match: PatternMatch
):
    # The up projection can occur after sigmoid/mul, so insert at the final mul.
    with graph_module.graph.inserting_before(match.anchor_node):
        fused = graph_module.graph.call_function(
            exir_ops.edge.et_vk.swiglu.default, tuple(match.input_nodes)
        )
    fused.meta = match.anchor_node.meta.copy()
    match.anchor_node.replace_all_uses_with(fused)
