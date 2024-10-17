# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult


class CastInt64ToInt32Pass(ExportPass):
    def __init__(self, exported_program: torch.export.ExportedProgram):
        super(CastInt64ToInt32Pass, self).__init__()
        self.exported_program = exported_program

    def _to_int32(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            fake_tensor = node.meta["val"]
            if isinstance(fake_tensor, torch._subclasses.fake_tensor.FakeTensor):
                if node.meta["val"].dtype == torch.int64:
                    node.meta["val"] = node.meta["val"].to(torch.int32)
                    buffer_name = (
                        self.exported_program.graph_signature.inputs_to_buffers[
                            node.name
                        ]
                    )
                    new_tensor = self.exported_program.state_dict[buffer_name].to(
                        torch.int32
                    )
                    self.exported_program.state_dict[buffer_name] = new_tensor

    def call(self, graph_module: torch.fx.GraphModule):
        self._to_int32(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
