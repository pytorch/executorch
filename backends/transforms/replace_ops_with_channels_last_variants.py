# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import ExportPass
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.node import Target
from torch.fx.passes.infra.pass_manager import PassResult

_NCHW_TO_NHWC_PERM: list[int] = [0, 2, 3, 1]
_NHWC_TO_NCHW_PERM: list[int] = [0, 3, 1, 2]


def _requires_rank(rank: int) -> Callable[[torch.fx.Node], bool]:
    def _inner(node: torch.fx.Node) -> bool:
        return len(node.meta["val"].shape) == rank

    return _inner


@dataclass
class ChannelsLastOpSpec:
    """Specification for replacing a contiguous op with its channels-last counterpart."""

    # The channels_last dialect target to replace the contiguous op with.
    target: Target

    # Positional arg indices of tensor inputs that should be permuted NCHW→NHWC.
    input_indices: list[int]

    # If provided, this function must return True for a node to be replaced.
    filter_fn: Callable[[torch.fx.Node], bool] | None = None

    # TODO Add output indices to support operators such as `max_pool2d_with_indices`.


_DEFAULT_OP_MAP: dict[Target, ChannelsLastOpSpec] = {
    exir_ops.edge.aten.convolution.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.convolution.default,
        input_indices=[0],
        filter_fn=_requires_rank(4),
    ),
    exir_ops.edge.aten.avg_pool2d.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.avg_pool2d.default,
        input_indices=[0],
        filter_fn=_requires_rank(4),
    ),
}


class ReplaceOpsWithChannelsLastVariants(ExportPass):
    """
    Replaces contiguous (NCHW) edge-dialect ops with their channels-last (NHWC)
    equivalents from the channels_last dialect, inserting permute_copy ops on
    the specified inputs and on the output so that the rest of the graph remains
    in contiguous layout.

    By default, all currently implemented channels_last dialect ops are replaced.
    Pass a custom op_map to restrict or extend the set of replacements.
    """

    def __init__(
        self,
        exported_program: ExportedProgram,
        op_map: dict[Target, ChannelsLastOpSpec] | None = None,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.op_map: dict[Target, ChannelsLastOpSpec] = (
            op_map if op_map is not None else dict(_DEFAULT_OP_MAP)
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if (spec := self.op_map.get(node.target)) is None:
                continue
            contiguous_dim_order = tuple(range(len(node.meta["val"].shape)))
            if node.meta["val"].dim_order() != contiguous_dim_order:
                continue
            if spec.filter_fn is not None and not spec.filter_fn(node):
                continue

            args = list(node.args)

            with graph.inserting_before(node):
                for idx in spec.input_indices:
                    perm_in = graph.create_node(
                        "call_function",
                        target=exir_ops.edge.channels_last.permute_copy.default,
                        args=(args[idx], _NCHW_TO_NHWC_PERM),
                    )
                    perm_in.meta = {}
                    args[idx] = perm_in

                nhwc_node = graph.create_node(
                    "call_function",
                    target=spec.target,
                    args=tuple(args),
                    kwargs=node.kwargs,
                )
                nhwc_node.meta = {}

                perm_out = graph.create_node(
                    "call_function",
                    target=exir_ops.edge.channels_last.permute_copy.default,
                    args=(nhwc_node, _NHWC_TO_NCHW_PERM),
                )
                perm_out.meta = {}

            node.replace_all_uses_with(perm_out)
            graph.erase_node(node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
