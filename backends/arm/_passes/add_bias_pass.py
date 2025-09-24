# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.transforms.utils import create_constant_placeholder

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind


class AddBiasPass(ArmPass):
    """TOSA requires convolution nodes to have a bias input.
    This pass adds a bias input to convolution nodes that do not have one.
    The bias is set to zero.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops = (exir_ops.edge.aten.convolution.default,)

    def call(self, graph_module):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in self.targeted_ops:
                continue

            if len(node.all_input_nodes) < 3:
                modified = True
                # bias is missing
                weight_node = node.all_input_nodes[1]
                output_channels = get_first_fake_tensor(weight_node).shape[0]
                # add a node containging zeros
                # if quantized, use int32, otherwise use float32
                if (
                    "output_qparams" in node.meta
                    and len(node.meta["output_qparams"]) > 0
                ):
                    bias_data = torch.zeros(size=(output_channels,), dtype=torch.int32)
                else:
                    bias_data = torch.zeros(
                        size=(output_channels,), dtype=torch.float32
                    )

                with graph_module.graph.inserting_after(weight_node):
                    bias_node = create_constant_placeholder(
                        self.exported_program,
                        graph=graph_module.graph,
                        kind=InputKind.PARAMETER,
                        data=bias_data,
                        persistent_buffer=True,
                        name=f"{node.name}_bias",
                    )
                    if node.args[0].meta["val"].dtype == torch.int16:
                        bias_node.meta[TosaSpecialDtype.meta_key()] = (
                            TosaSpecialDtype.INT48
                        )
                node.update_arg(2, bias_node)

        if modified:
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
