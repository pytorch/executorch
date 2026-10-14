# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from math import prod

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_overload_packet,
    get_shape,
)
from executorch.backends.transforms.quantize_reorder_utils import (
    slice_or_select_overloadpkt,
    trivially_quantizable_ops_overloadpkt,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.tensor import num_bytes_from_shape_and_dtype


class PostponeDequantizeOpBelowUseChainPass(ExportPass):
    """
    If the consumer of dequantize is a linear chain of view, transpose, permute,
    or slice ops that are trivially quantized, we can convert the pattern
    dequantize(int8/uint8) -> view/transpose/permute/slice(fp32) to
    view/transpose/permute/slice(int8/uint8) -> dequantize(int8/uint8)
    The benefit of such reordering is that the view/transpose/permute/slice
    will move far less data.
    """

    quantize_op_packets: set[object] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_channel,
    }
    dequantize_packet_to_overload: dict[object, str] = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor: "default",
        exir_ops.edge.quantized_decomposed.dequantize_per_channel: "default",
    }

    def __init__(self):
        super().__init__()
        self.graph_module = None

    # Return true if postponing the dequantize node is feasible
    def postponing_feasible(self, dequant_node: torch.fx.Node):
        users = list(dequant_node.users.keys())
        # Check if the dequantize op has a single user, and that user is
        # trivially quantizable.
        trivially_quantizable_users = all(
            get_overload_packet(user.target) in trivially_quantizable_ops_overloadpkt
            for user in users
        )
        if len(users) == 1:
            return trivially_quantizable_users

        # Otherwise check if all the users are slice op
        if not all(
            get_overload_packet(user.target) in slice_or_select_overloadpkt
            for user in users
        ):
            return False

        quantized_branches = []
        for user in users:
            slice_users = list(user.users)
            quantized_branches.append(
                bool(slice_users)
                and all(
                    slice_user.op == "call_function"
                    and get_overload_packet(slice_user.target)
                    in self.quantize_op_packets
                    for slice_user in slice_users
                )
            )

        # Preserve the existing fallback for forks whose branches all requantize.
        if all(quantized_branches):
            return True

        dequant_shape = get_shape(self.graph_module, dequant_node)
        if dequant_shape is None:
            return False

        dequant_bytes = num_bytes_from_shape_and_dtype(dequant_shape, torch.float32)
        surviving_dequant_bytes = 0
        for user, quantized_branch in zip(users, quantized_branches):
            if quantized_branch:
                continue
            slice_shape = get_shape(self.graph_module, user)
            if slice_shape is None:
                return False
            # Nop slices are removed later and do not add another materialized copy.
            if prod(list(slice_shape)) == prod(list(dequant_shape)):
                continue
            surviving_dequant_bytes += num_bytes_from_shape_and_dtype(
                slice_shape, torch.float32
            )

        return surviving_dequant_bytes <= dequant_bytes

    def postpone_dequantize_op(self, graph_module: torch.fx.GraphModule) -> bool:
        packet_to_overload_map = self.dequantize_packet_to_overload
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            overload_packet = get_overload_packet(node.target)
            if (
                overload_packet not in packet_to_overload_map.keys()
                or not self.postponing_feasible(node)
            ):
                continue

            for user in node.users:
                with graph.inserting_after(user):
                    dequant_node = graph.call_function(
                        getattr(
                            overload_packet, packet_to_overload_map[overload_packet]
                        ),
                        args=(user, *node.args[1:]),
                    )
                    dequant_node.meta = user.meta.copy()
                    # Remove meta["debug_handle"] on new node if it exists.
                    # Reassign it at the caller level by calling generate_missing_debug_handles
                    dequant_node.meta.pop("debug_handle", None)
                    user.replace_all_uses_with(dequant_node)
                    dequant_node.args = (user, *node.args[1:])

            pred = node.args[0]
            node.replace_all_uses_with(pred)
            graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # The logic in postpone_dequantize_op that handles branching checks the shape
        # of the dequant node, which isn't available if that node was already postponed
        # in the same pass invokation. The shape information is recreated by tracing in
        # super().call(), meaning that every branch in the graph that we wish to postpone
        # dequant past requires retracing. We iterate the pass until it no longer modifies
        # the graph (up to 3 times max, to avoid potential infinite loops)
        self.graph_module = graph_module
        iter_count = 0
        local_modified = False
        overall_modified = False

        while local_modified or iter_count == 0:
            local_modified = self.postpone_dequantize_op(self.graph_module)
            overall_modified |= local_modified

            if local_modified:
                self.graph_module = super().call(self.graph_module).graph_module

            iter_count += 1
            if iter_count == 3:
                break

        return PassResult(self.graph_module, overall_modified)
