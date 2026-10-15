# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import get_edge_overload_packet
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import PassResult
from torch.fx.passes.infra.pass_base import PassBase


class RemoveBranchedQuantDequant(PassBase):
    """
    This pass looks for adjacent quant and dequant nodes with identical
    parameters, where the quant node has other users in addition to the
    dequant. The quant and dequant pair would be removed by the
    FuseQuantDequantToRequantizePass if not for the multiple users. This pass
    removes just the dequant node by connecting it to the quant's parent node
    """

    quantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
    }
    dequantize_op_packets: set[EdgeOpOverloadPacket] = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = self.remove_branched(
            graph_module, self.quantize_op_packets, self.dequantize_op_packets
        )
        modified |= self.remove_branched(
            graph_module, self.dequantize_op_packets, self.quantize_op_packets
        )

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)

    def remove_branched(
        self,
        graph_module: torch.fx.GraphModule,
        producer_pkts: set[EdgeOpOverloadPacket],
        consumer_pkts: set[EdgeOpOverloadPacket],
    ) -> bool:
        modified = False
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or not isinstance(node.target, EdgeOpOverload)
                or get_edge_overload_packet(node.target) not in producer_pkts
            ):
                continue

            if len(node.users) < 2:
                continue

            for user in node.users:
                if (
                    not isinstance(user.target, EdgeOpOverload)
                    or get_edge_overload_packet(user.target) not in consumer_pkts
                ):
                    continue

                # check qparams match
                if node.args[1:] != user.args[1:]:
                    continue

                user.replace_all_uses_with(node.args[0])
                modified = True

        return modified
