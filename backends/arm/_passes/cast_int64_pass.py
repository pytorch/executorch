# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging

import torch
from executorch.backends.arm._passes.arm_pass_utils import is_param_node
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import is_buffer

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class CastInt64ToInt32Pass(ExportPass):
    def __init__(self, exported_program: torch.export.ExportedProgram):
        super(CastInt64ToInt32Pass, self).__init__()
        self.exported_program = exported_program

    def _to_int32(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            fake_tensor = node.meta["val"]
            if isinstance(fake_tensor, torch._subclasses.fake_tensor.FakeTensor):
                if node.meta["val"].dtype == torch.int64 and is_param_node(
                    self.exported_program, node
                ):
                    if is_buffer(self.exported_program, node):
                        node.meta["val"] = node.meta["val"].to(torch.int32)
                        buffer_name = (
                            self.exported_program.graph_signature.inputs_to_buffers[
                                node.name
                            ]
                        )
                        buffer = self.exported_program.state_dict[node.name]
                        logger.warning(
                            f"Casting buffer {node.name} from torch.int64 to torch.int32"
                            f" defined in {node.meta['stack_trace']}"
                        )
                        if torch.min(buffer) < torch.iinfo(torch.int32).min:
                            raise RuntimeError(
                                f"Buffer {node.name} has value < {torch.iinfo(torch.int32).min}"
                            )
                        if torch.max(buffer) > torch.iinfo(torch.int32).max:
                            raise RuntimeError(
                                f"Buffer {node.name} has value > {torch.iinfo(torch.int32).max}"
                            )
                        buffer_int32 = buffer.to(torch.int32)
                        self.exported_program.state_dict[buffer_name] = buffer_int32

    def call(self, graph_module: torch.fx.GraphModule):
        self._to_int32(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
