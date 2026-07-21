# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from dataclasses import dataclass
from typing import Callable

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import torch

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass
from torch.fx.node import Target
from torch.fx.passes.infra.pass_manager import PassResult

_NCHW_TO_NHWC_PERM: list[int] = [0, 2, 3, 1]
_NHWC_TO_NCHW_PERM: list[int] = [0, 3, 1, 2]


FilterFnType = Callable[[torch.fx.Node], bool]


def _requires_rank(allowed_ranks: list[int]) -> FilterFnType:
    def _inner(node) -> bool:
        val = node.meta["val"]
        val = val[0] if isinstance(val, (list, tuple)) else val
        return val.dim() in allowed_ranks

    return _inner


def _has_2d_convolution_kernel() -> FilterFnType:
    def _inner(node) -> bool:
        # Extract kernel size from the weights
        try:
            w = node.args[1]
            return w.meta["val"].dim() == 4
        except AttributeError:
            return False

    return _inner


def _and(*filter_fns: FilterFnType) -> FilterFnType:
    def _inner(*args) -> bool:
        return all(fn(*args) for fn in filter_fns)

    return _inner


@dataclass
class ChannelsLastOpSpec:
    """Specification for replacing a contiguous op with its channels-last counterpart."""

    # The channels_last dialect target to replace the contiguous op with.
    target: Target

    # Positional arg indices of tensor inputs that should be permuted NCHW→NHWC.
    input_indices: list[int]

    # Indices of the outputs that should be permuted NCHW→NHWC.
    output_indices: list[int]

    # If provided, this function must return True for a node to be replaced.
    filter_fn: FilterFnType | None = None


_DEFAULT_OP_MAP: dict[Target, ChannelsLastOpSpec] = {
    exir_ops.edge.aten._adaptive_avg_pool2d.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.adaptive_avg_pool2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_requires_rank([3, 4]),
    ),
    exir_ops.edge.aten.avg_pool2d.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.avg_pool2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_requires_rank([3, 4]),
    ),
    exir_ops.edge.aten.convolution.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.convolution.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_and(_requires_rank([3, 4]), _has_2d_convolution_kernel()),
    ),
    exir_ops.edge.aten.grid_sampler_2d.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.grid_sampler_2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_requires_rank([4]),
    ),
    exir_ops.edge.aten.max_pool2d_with_indices.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.max_pool2d_with_indices.default,
        input_indices=[0],
        output_indices=[0, 1],
        filter_fn=_requires_rank([3, 4]),
    ),
    exir_ops.edge.aten.upsample_bilinear2d.vec: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.upsample_bilinear2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_requires_rank([4]),
    ),
    exir_ops.edge.aten.upsample_nearest2d.vec: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.upsample_nearest2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_requires_rank([4]),
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

    @staticmethod
    def _permute_node_input(
        graph: torch.fx.Graph, node_input: torch.fx.Node, implicit_batch: bool
    ) -> torch.fx.Node:
        if implicit_batch:
            # Explicitly add batch size of `1`.
            node_input = graph.create_node(
                "call_function",
                target=exir_ops.edge.aten.unsqueeze_copy.default,
                args=(node_input, 0),
            )
            node_input.meta = {}

        res = graph.create_node(
            "call_function",
            target=exir_ops.edge.channels_last.permute_copy.default,
            args=(node_input, _NCHW_TO_NHWC_PERM),
        )
        res.meta = {}

        return res

    @staticmethod
    def _permute_node_output(
        graph: torch.fx.Graph,
        node_output: torch.fx.Node,
        original_node_output: torch.fx.Node,
        implicit_batch: bool,
    ):
        output = graph.create_node(
            "call_function",
            target=exir_ops.edge.channels_last.permute_copy.default,
            args=(node_output, _NHWC_TO_NCHW_PERM),
        )
        output.meta = {}

        if implicit_batch:
            # Remove the explicitly added batch size of `1`.
            output = graph.create_node(
                "call_function",
                target=exir_ops.edge.aten.squeeze_copy.dims,
                args=(output, [0]),
            )
            output.meta = {}

        original_node_output.replace_all_uses_with(output)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if (spec := self.op_map.get(node.target)) is None:
                continue
            val = node.meta["val"]
            val = val[0] if isinstance(val, (list, tuple)) else val
            contiguous_dim_order = tuple(range(val.dim()))
            if val.dim_order() != contiguous_dim_order:
                continue
            if spec.filter_fn is not None and not spec.filter_fn(node):
                continue

            # In case of implicit batch size, insert also `unsqueeze_copy.default` and `squeeze_copy.dims` operators.
            #  With `convolution`, this already happens during lowering to edge. But it doesn't happen for example with
            #  `avg_pool2d`. Therefore, we do it manually here to achieve consistency between the operators.
            implicit_batch = val.dim() == 3

            args = list(node.args)

            with graph.inserting_before(node):
                for idx in spec.input_indices:
                    args[idx] = self._permute_node_input(
                        graph, args[idx], implicit_batch
                    )

                nhwc_node = graph.create_node(
                    "call_function",
                    target=spec.target,
                    args=tuple(args),
                    kwargs=node.kwargs,
                )
                nhwc_node.meta = {}

                users = list(node.users)
                if all(
                    u.op == "call_function" and u.target == operator.getitem
                    for u in users
                ):
                    # `node` produces multiple outputs which are extracted by following `getitem` nodes.
                    for idx in spec.output_indices:
                        old_getitem_nodes = [u for u in users if u.args[1] == idx]
                        if not old_getitem_nodes:
                            continue  # The output is not used.

                        # Add a new `getitem` node.
                        new_getitem_node = graph.create_node(
                            "call_function",
                            target=operator.getitem,
                            args=(nhwc_node, idx),
                        )
                        new_getitem_node.meta = {}

                        self._permute_node_output(
                            graph,
                            new_getitem_node,
                            old_getitem_nodes[0],
                            implicit_batch,
                        )

                        # Remove the old `getitem` node.
                        graph.erase_node(old_getitem_nodes[0])
                else:
                    # Regular node with a single output.
                    assert spec.output_indices == [
                        0
                    ], f"Incorrect use of `output_indices` for op `{spec.target}`."
                    self._permute_node_output(graph, nhwc_node, node, implicit_batch)

            graph.erase_node(node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
