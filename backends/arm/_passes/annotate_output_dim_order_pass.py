# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_output_dim_orders
from executorch.exir.pass_base import PassResult


class AnnotateOutputDimOrderPass(ArmPass):
    """
    Stores the current output dim_orders in the meta dict of the output node. This is used
    for verifying that the dim order does not change unexpectedly in later passes.
    """

    def call(self, graph_module):
        output_node = graph_module.graph.output_node()
        output_node.meta["original_dim_orders"] = get_output_dim_orders(graph_module)

        return PassResult(graph_module, True)
