# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, Optional

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    FuseOpPairsAcrossBranchesPass,
    get_arg,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import PassResult
from torch.utils import _pytree as pytree


class FuseQuantDequantToRequantizePass(FuseOpPairsAcrossBranchesPass):
    """
    Fuse dequantize-quantize op pairs to a single requantize op.
    For the special case where quant params match, this will remove
    both dequant and quant ops.
    """

    # A list of ops that can be bypassed when looking for a
    # dequantize->quantize chain
    bypass_ops: set[EdgeOpOverload] = {
        exir_ops.edge.aten.slice_copy.Tensor,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten.transpose_copy.int,
        exir_ops.edge.aten.permute_copy.default,
    }

    quantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
    }
    dequantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
    }
    # Op used to fuse a dequantize-quantize pair whose qparams differ. There is
    # no backend-neutral requantize op, so a backend that wants such pairs fused
    # must set this; otherwise only pairs with matching qparams are removed.
    requantize_op: Optional[EdgeOpOverload] = None

    def __init__(
        self, allow_requantize: bool = False, force_quant_dequant_fusion: bool = False
    ) -> None:
        super().__init__()
        if allow_requantize and self.requantize_op is None:
            raise ValueError(
                f"{type(self).__name__} has no requantize_op, so it cannot "
                "run with allow_requantize=True"
            )
        self.allow_requantize: bool = allow_requantize
        self.force_quant_dequant_fusion: bool = force_quant_dequant_fusion

    def _pkg_name_match(self, node1: torch.fx.Node, node2: torch.fx.Node) -> bool:
        # pyre-ignore[16]: Item `typing.Callable` has no attribute `_op`
        return node1.target._op.namespace == node2.target._op.namespace

    def can_fuse_for_chain(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        consumer_op_packets: set[EdgeOpOverloadPacket],
    ) -> bool:
        return super().can_fuse_for_chain(
            producer, consumer, consumer_op_packets
        ) and self._pkg_name_match(producer, consumer)

    def _create_requantize_node(
        self,
        in_tensor: torch.fx.Node,
        in_scale: float,
        in_zero_point: int,
        out_scale: float,
        out_zero_point: int,
        out_dtype: torch.dtype,
        graph: torch.fx.Graph,
    ) -> torch.fx.Node:
        return graph.call_function(
            cast(EdgeOpOverload, self.requantize_op),
            args=(
                in_tensor,
                in_scale,
                in_zero_point,
                out_scale,
                out_zero_point,
                out_dtype,
            ),
        )

    def _quant_params_match(self, node1: torch.fx.Node, node2: torch.fx.Node) -> bool:
        arg_names = ("scale", "zero_point", "quant_min", "quant_max", "dtype")
        arg_types = (float, int, int, int, torch.dtype)
        return all(
            get_arg(node1, name, arg_type) == get_arg(node2, name, arg_type)
            for name, arg_type in zip(arg_names, arg_types)
        )

    def check_ok_to_fuse(
        self,
        producer: torch.fx.Node,
        consumers: list[torch.fx.Node],
    ) -> bool:
        """Check if all node-user pairs are nops or are ok to replace with requant."""
        for rnode in consumers:
            if self.allow_requantize or self._quant_params_match(producer, rnode):
                # Cannot remove quant-dequant pair if quant params don't match and requantize
                # is not allowed.
                continue
            return False
        return True

    def _get_bypassed_nodes(
        self,
        producer: torch.fx.Node,
        removal_candidates: list[torch.fx.Node],
    ) -> list[torch.fx.Node]:
        """Return the bypassed ops between producer and consumers, in order."""
        candidates = set(removal_candidates)
        bypassed: set[torch.fx.Node] = set()
        stack = list(producer.users)
        while stack:
            user = stack.pop()
            if user in candidates or user in bypassed:
                continue
            bypassed.add(user)
            stack.extend(user.users)

        # The graph's node list is topologically sorted, so walking it forward
        # from the producer refreshes each node after its predecessor.
        ordered: list[torch.fx.Node] = []
        cursor = producer.next
        while bypassed and cursor.op != "root":
            if cursor in bypassed:
                bypassed.remove(cursor)
                ordered.append(cursor)
            cursor = cursor.next
        return ordered

    def fuse(
        self,
        node: torch.fx.Node,
        removal_candidates: list[torch.fx.Node],
        graph_module: torch.fx.GraphModule,
    ) -> None:
        # Capture the bypassed ops before the rewire erases the chain's root.
        bypassed_nodes = self._get_bypassed_nodes(node, removal_candidates)
        node.replace_all_uses_with(cast(torch.fx.Node, node.args[0]))
        graph_module.graph.erase_node(node)
        # They now consume the producer's input, so their metadata describes an
        # operand they no longer have. Refresh before get_fused_node reads it.
        for bypassed in bypassed_nodes:
            args, kwargs = pytree.tree_map_only(
                torch.fx.Node,
                lambda arg: arg.meta["val"],
                (bypassed.args, bypassed.kwargs),
            )
            assert callable(bypassed.target)
            bypassed.meta["val"] = bypassed.target(*args, **kwargs)
            bypassed.meta["tensor_meta"] = None
        for rnode in removal_candidates:
            rnode.replace_all_uses_with(self.get_fused_node(node, rnode, graph_module))
            graph_module.graph.erase_node(rnode)

    def get_fused_node(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        graph_module: torch.fx.GraphModule,
    ) -> torch.fx.Node:
        in_scale = get_arg(producer, "scale", float)
        in_zero_point = get_arg(producer, "zero_point", int)
        out_scale = get_arg(consumer, "scale", float)
        out_zero_point = get_arg(consumer, "zero_point", int)
        out_dtype = get_arg(consumer, "dtype", torch.dtype)
        if in_scale == out_scale and in_zero_point == out_zero_point:
            # If the quant params match, we can remove both dequantize-quantize ops.
            return cast(torch.fx.Node, consumer.args[0])

        assert self.allow_requantize, (
            f"Found {producer=} {in_scale=} {in_zero_point=} | {consumer=} {out_scale=} {out_zero_point=}"
        )

        with graph_module.graph.inserting_before(consumer):
            requantize_node = self._create_requantize_node(
                in_tensor=cast(torch.fx.Node, consumer.args[0]),
                in_scale=in_scale,
                in_zero_point=in_zero_point,
                out_scale=out_scale,
                out_zero_point=out_zero_point,
                out_dtype=out_dtype,
                graph=graph_module.graph,
            )
            requantize_node.meta = consumer.meta.copy()
            requantize_node.meta["val"] = cast(EdgeOpOverload, self.requantize_op)(
                cast(torch.fx.Node, consumer.args[0]).meta["val"],
                cast(float, in_scale),
                cast(int, in_zero_point),
                cast(float, out_scale),
                cast(int, out_zero_point),
                cast(torch.dtype, out_dtype),
            )
            requantize_node.meta["tensor_meta"] = None
        return requantize_node

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Remove any dequantize op that has only quantize ops as its users.
        modified = self.find_and_fuse(
            graph_module,
            producer_op_packets=self.dequantize_op_packets,
            consumer_op_packets=self.quantize_op_packets,
            bypass_ops=self.bypass_ops,
        )
        # Remove any quantize op that has only dequantze ops as its users.
        modified |= self.find_and_fuse(
            graph_module,
            producer_op_packets=self.quantize_op_packets,
            consumer_op_packets=self.dequantize_op_packets,
            # Do not requantize for quantize-dequantize pairs as this is not guaranteed
            # to be better for performance/memory.
            # Only fuse if all users of quant are dequant.
            bypass_ops=(
                self.bypass_ops
                if self.force_quant_dequant_fusion
                else {exir_ops.edge.aten.view_copy.default}
            ),
        )
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            return PassResult(graph_module, True)
        return PassResult(graph_module, False)
