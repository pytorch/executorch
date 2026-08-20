# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, Callable, cast

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    FuseOpPairsAcrossBranchesPass,
    get_permuted_dims,
    get_transposed_dims,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import PassResult


class FuseTransposeOrPermuteOpPairsPass(FuseOpPairsAcrossBranchesPass):
    """
    Fuse transpose or permute op pairs to a single view op.
    (transpose or permutation) -> (quant or dequant) -> (transpose or permutation)
    This happens when op2(op1) == identity, modulo unitary dimensions.
    'unitary dimensions' example: a tensor of shape [1, 5, 30] is equivalent (in memory) to [5, 1, 30]
    so transpose(1, 2) then transpose(0, 2) is a pseudo identity and should be fused.
    """

    # A list of ops that can be bypassed when looking for a
    # transpose-permute chain. Subclasses can extend this with backend-specific ops.
    bypass_ops: set[EdgeOpOverload] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    }

    def can_fuse_for_chain(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        consumer_op_packets: set[EdgeOpOverloadPacket],
    ) -> bool:
        if not super().can_fuse_for_chain(producer, consumer, consumer_op_packets):
            return False

        # checking that permut2(permut1(identity)) == identity, modulo unitary dimensions
        producer_input = cast(torch.fx.Node, producer.args[0])
        if "val" not in producer_input.meta:
            return False
        input_shape = producer_input.meta["val"].shape
        ident_dims = list(range(len(input_shape)))
        # this mapping helps to handle both transpose and permutations
        f: dict[Any, Callable] = {
            exir_ops.edge.aten.transpose_copy.int: get_transposed_dims,
            exir_ops.edge.aten.permute_copy.default: get_permuted_dims,
        }
        in_dims = f[producer.target](producer, ident_dims)
        out_dims = f[consumer.target](consumer, in_dims)
        # Filtering out unitary dimensions
        non_unit_ident_dims = [dim for dim in ident_dims if input_shape[dim] != 1]
        non_unit_out_dims = [dim for dim in out_dims if input_shape[dim] != 1]
        return non_unit_out_dims == non_unit_ident_dims

    def get_fused_node(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        graph_module: torch.fx.GraphModule,
    ) -> torch.fx.Node:
        # This step is important because of how we can fuse transpositions that are not perfectly
        # reverse one of another but will be fused if there are unitary dimensions.
        # The fused operation must have the same output shape as the consumer.
        output_shape = consumer.meta["val"].shape
        with graph_module.graph.inserting_after(consumer):
            view = graph_module.graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                (consumer.args[0], output_shape),
                {},
            )
        return view

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Remove any transpose/permutation op pair that cancel each other.
        modified = self.find_and_fuse(
            graph_module,
            producer_op_packets={
                exir_ops.edge.aten.transpose_copy,
                exir_ops.edge.aten.permute_copy,
            },
            consumer_op_packets={
                exir_ops.edge.aten.transpose_copy,
                exir_ops.edge.aten.permute_copy,
            },
            bypass_ops=self.bypass_ops,
        )
        if modified:
            return super().call(graph_module)
        return PassResult(graph_module, False)
