# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.backends.fused_quant.ops  # noqa
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassBase, PassResult
from torch import fx
from torch._ops import OpOverload


# Volume-preserving data-movement ops that only rearrange tensor layout
# without changing element values.
_DATA_MOVEMENT_OPS: frozenset[OpOverload] = frozenset(
    {
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.view_copy.default,
    }
)

# The elementwise activations this pass hoists.
_ACTIVATION_OPS: frozenset[OpOverload] = frozenset(
    {
        exir_ops.edge.fused_quant.relu.default,
        exir_ops.edge.fused_quant.hardswish.default,
        exir_ops.edge.fused_quant.sigmoid.default,
        exir_ops.edge.fused_quant.tanh.default,
        exir_ops.edge.fused_quant.hard_tanh.default,
        exir_ops.edge.fused_quant.silu.default,
        exir_ops.edge.fused_quant.hardsigmoid.default,
        exir_ops.edge.fused_quant.gelu.default,
    }
)


class HoistActivations(PassBase):
    """Hoist element-wise activations above the data-movement ops feeding them.

    Turns chains like ``permute → relu`` into ``relu → permute`` by moving each
    single-user data-movement op forward past the activation. Each relocated
    data-movement op is retyped to the activation's output dtype.
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph

        for target in _ACTIVATION_OPS:
            for activation_node in graph.find_nodes(op="call_function", target=target):
                out_dtype = activation_node.meta["val"].dtype

                data_movement_node = activation_node.args[0]
                while (
                    data_movement_node.target in _DATA_MOVEMENT_OPS
                    and len(data_movement_node.users) == 1
                ):
                    input_node = data_movement_node.args[0]

                    # Rewire connections:
                    # from: input_node -> data_movement_node -> activation_node -> consumers
                    #   to: input_node -> activation_node -> data_movement_node -> consumers
                    activation_node.replace_all_uses_with(data_movement_node)
                    data_movement_node.replace_input_with(input_node, activation_node)
                    activation_node.replace_input_with(data_movement_node, input_node)

                    # Move data_movement_node to sit right after activation_node.
                    # Its sole user is activation_node, so this is always valid.
                    activation_node.append(data_movement_node)

                    # activation_node keeps its own output dtype but takes its new
                    # input's shape; data_movement_node keeps its shape but now
                    # carries activation_node's output dtype.
                    activation_node.meta["val"] = input_node.meta["val"].to(out_dtype)
                    data_movement_node.meta["val"] = data_movement_node.meta["val"].to(
                        out_dtype
                    )
                    # Advance to the next candidate: whatever now feeds activation_node.
                    data_movement_node = activation_node.args[0]
                    modified = True

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)
