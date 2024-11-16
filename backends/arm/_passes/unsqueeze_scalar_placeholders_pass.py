# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.pass_base import ExportPass, PassResult


class UnsqueezeScalarPlaceholdersPass(ExportPass):
    """
    Placeholders that have node.meta["val"].shape = () cause issues later in the lowering.
    This pass unsqueezes the placeholders to make sure shape is at least (1,).
    """

    def __init__(self, exported_program):
        self.exported_program = exported_program
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            rank = node.meta["val"].dim()
            if rank == 0:
                if not (
                    node.name in self.exported_program.graph_signature.inputs_to_buffers
                    or node.name
                    in self.exported_program.graph_signature.inputs_to_parameters
                ):
                    continue
                tensor = self.exported_program.state_dict[node.name]
                if tensor.dim() == 0:
                    self.exported_program.state_dict[node.name] = tensor.unsqueeze(0)
                    node.meta["val"] = node.meta["val"].fake_mode.from_tensor(
                        tensor.unsqueeze(0), static_shapes=True
                    )
                else:
                    node.meta["val"] = node.meta["val"].fake_mode.from_tensor(
                        tensor, static_shapes=True
                    )

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)

    def ensures(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                rank = node.meta["val"].dim()
                if rank == 0:
                    raise ValueError("Placeholders of rank 0 are not supported!")
