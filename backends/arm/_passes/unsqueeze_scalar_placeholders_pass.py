# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import is_buffer, is_param


class UnsqueezeScalarPlaceholdersPass(ArmPass):
    """
    Placeholders that have node.meta["val"].shape = () cause issues later in the lowering.
    This pass unsqueezes the placeholders to make sure shape is at least (1,).
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            rank = node.meta["val"].dim()
            if rank == 0:
                if is_buffer(self.exported_program, node):
                    name = self.exported_program.graph_signature.inputs_to_buffers[
                        node.name
                    ]
                elif is_param(self.exported_program, node):
                    name = self.exported_program.graph_signature.inputs_to_parameters[
                        node.name
                    ]
                else:
                    continue

                tensor = self.exported_program.state_dict[name]

                if tensor.dim() == 0:
                    self.exported_program.state_dict[name] = tensor.unsqueeze(0)
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
                    if not (
                        is_buffer(self.exported_program, node)
                        or is_param(self.exported_program, node)
                    ):
                        continue
                    raise ValueError("Placeholders of rank 0 are not supported!")
