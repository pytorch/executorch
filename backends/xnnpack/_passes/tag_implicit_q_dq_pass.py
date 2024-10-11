# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List, Optional

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.partition.configs import (
    SUPPORTED_IMPLICIT_Q_DQ_MODULES_SET,
    SUPPORTED_IMPLICIT_Q_DQ_OP_NAMES_SET,
)
from executorch.backends.xnnpack.utils.quant_utils import (
    is_dequant,
    is_dynamic_qdq,
    is_quant,
)
from executorch.backends.xnnpack.utils.utils import is_param_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class TagImplicitQDqPass(XNNPACKPass):
    """
    This pass is used to tag "implicit" q/dq nodes, which should be ignored
    during preprocessing.

    A q or dq node is deemed to be "implicit" if any of the following hold:
    a) All of its inputs are constants (get_attr nodes or parameter (placeholder) nodes),
       since (de)quantizing constants is done outside of executing the graph
    b) It is the q or dq surrounding a "supported" group of nodes, ordered as
       dq -> [supported group] -> q. A "supported" group is comprised of one of
       the following:
       (  i) A single supported op, from SUPPORTED_QUANT_OPS_SET,
       ( ii) A single supported module, from SUPPORTED_QUANT_MODULES_SET, or
       (iii) a chain of nodes matching a supported chain from
             SUPPORTED_QUANT_CHAINS.
       q/dq nodes which match this condition should be
       ignore during preprocessing because they are only used as signaling for q
       params of node inputs
    c) It is a dq followed by aten.linear.default and then an output node. This
       is because aten.linear.default is a special op corresponding with
       dqlinear which doesn't necessarily have an q after it
    """

    _END_OF_CHAIN_MARKER = "END_OF_CHAIN"
    # TODO: @salilsdesai Avoid hardcoding quant module chains here (instead get from quantizer)
    SUPPORTED_QUANT_CHAINS = {
        exir_ops.edge.aten.add.Tensor.name(): {
            exir_ops.edge.aten.relu.default.name(): {
                _END_OF_CHAIN_MARKER: True,
            }
        },
        exir_ops.edge.aten.convolution.default.name(): {
            exir_ops.edge.aten.relu.default.name(): {
                _END_OF_CHAIN_MARKER: True,
            }
        },
        exir_ops.edge.aten.mul.Tensor.name(): {
            exir_ops.edge.aten.relu.default.name(): {
                _END_OF_CHAIN_MARKER: True,
            }
        },
        exir_ops.edge.aten.sub.Tensor.name(): {
            exir_ops.edge.aten.relu.default.name(): {
                _END_OF_CHAIN_MARKER: True,
            }
        },
        exir_ops.edge.aten.linear.default.name(): {
            exir_ops.edge.aten.relu.default.name(): {
                _END_OF_CHAIN_MARKER: True,
            }
        },
    }
    IS_IMPLICIT_Q_DQ_TAG = "IS_IMPLICIT_Q_DQ_TAG"

    def is_output_node(self, node: torch.fx.Node) -> bool:
        return node.op == "output"

    def is_dynamically_quantized(self, node: torch.fx.Node) -> bool:
        return is_dynamic_qdq(node)

    def is_supported_quant_op(self, node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and cast(torch._ops.OpOverload, node.target).name()
            in SUPPORTED_IMPLICIT_Q_DQ_OP_NAMES_SET
        )

    def is_supported_quant_module(self, node: torch.fx.Node) -> bool:
        is_supported = (
            "source_fn_stack" in node.meta
            and node.meta["source_fn_stack"][-1][1]
            in SUPPORTED_IMPLICIT_Q_DQ_MODULES_SET
        )
        if is_supported and self.is_supported_quant_op(node):
            raise RuntimeError(
                f"The same node should not be both a supported quant op and supported quant module: {node}"
            )
        return is_supported

    def tag_as_implicit_q_dq(self, node: torch.fx.Node) -> None:
        node.meta[TagImplicitQDqPass.IS_IMPLICIT_Q_DQ_TAG] = True

    @staticmethod
    def is_tagged_as_implicit_q_dq(node: torch.fx.Node) -> bool:
        return node.meta.get(TagImplicitQDqPass.IS_IMPLICIT_Q_DQ_TAG, False)

    def get_ending_implicit_q_nodes(
        self, start_node: torch.fx.Node
    ) -> Optional[List[torch.fx.Node]]:
        """
        Returns a list of implicit q nodes which end the potential "supported"
        group of nodes starting with start_node (which came after a dq), or None
        if no such "supported" group exists. This list will either contain
        one or zero elements.
        """
        # If the node after the dq has multiple users then the dq can't be
        # implicit
        if len(start_node.users) != 1:
            return None

        next_node = list(start_node.users)[0]

        if is_quant(next_node):
            # Check if second_node (which is between dq and q nodes) is in
            # supported quant ops or modules set
            if self.is_supported_quant_op(start_node) or self.is_supported_quant_module(
                start_node
            ):
                return [next_node]
        elif self.is_output_node(next_node):
            # if node following dq is output node
            return None
        else:
            # Check if nodes between the dq node and the next q match
            # a supported quant chain
            available_chains = TagImplicitQDqPass.SUPPORTED_QUANT_CHAINS
            current_node = start_node
            while (
                # Not yet at end of chain in graph
                not is_quant(current_node)
                # Right number of users to continue chain
                and len(current_node.users) == 1
                # Can continue following an available chain
                and (
                    current_node.op == "call_function"
                    and cast(torch._ops.OpOverload, current_node.target).name()
                    in available_chains
                )
            ):
                available_chains = available_chains[
                    cast(torch._ops.OpOverload, current_node.target).name()
                ]
                current_node = list(current_node.users)[0]

            if (
                is_quant(current_node)
                and TagImplicitQDqPass._END_OF_CHAIN_MARKER in available_chains
            ):
                # The chain of nodes between the dq and q nodes matches
                # a supported quant chain
                return [current_node]

        return None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for first_node in graph_module.graph.nodes:
            if (is_dequant(first_node) or is_quant(first_node)) and all(
                is_param_node(self.exported_program, n)
                for n in first_node.all_input_nodes
            ):
                # All of the q or dq node's inputs are constants
                self.tag_as_implicit_q_dq(first_node)
                continue

            if not is_dequant(first_node):
                continue

            if len(first_node.users) == 0:
                continue

            ending_implicit_q_nodes = []
            for user in first_node.users:
                if self.is_dynamically_quantized(first_node):
                    # if the dq is a dynamic dq, then it is implicit
                    break
                user_end_nodes = self.get_ending_implicit_q_nodes(user)
                if user_end_nodes is None:
                    # This user isn't part of a "supported" group
                    ending_implicit_q_nodes = None
                    break
                ending_implicit_q_nodes.extend(user_end_nodes)

            if ending_implicit_q_nodes is None:
                # There was a user which isn't part of a "supported" group
                # Don't tag anything as implicit for this iteration
                continue

            self.tag_as_implicit_q_dq(first_node)
            for node in ending_implicit_q_nodes:
                self.tag_as_implicit_q_dq(node)

        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
