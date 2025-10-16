# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import is_buffer
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)


class CastInt64BuffersToInt32Pass(ArmPass):
    """
    Cast int64 buffers to int32 if the int64 data is in int32 range.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram):
        super().__init__()
        self.exported_program = exported_program

    def _assert_within_int32(self, tensor: torch.Tensor, node: torch.fx.Node):
        if torch.min(tensor) < torch.iinfo(torch.int32).min:
            raise RuntimeError(
                f"Node {node.name} has value < {torch.iinfo(torch.int32).min}"
            )
        if torch.max(tensor) > torch.iinfo(torch.int32).max:
            raise RuntimeError(
                f"Node {node.name} has value > {torch.iinfo(torch.int32).max}"
            )

    def _to_int32(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if len(node.users) == 0:
                continue
            fake_tensor = node.meta["val"]
            if not isinstance(fake_tensor, torch._subclasses.fake_tensor.FakeTensor):
                continue
            if fake_tensor.dtype != torch.int64:
                continue
            if is_buffer(self.exported_program, node):
                node.meta["val"] = fake_tensor.to(torch.int32)
                buffer_name = self.exported_program.graph_signature.inputs_to_buffers[
                    node.name
                ]
                buffer = self.exported_program.state_dict[buffer_name]
                self._assert_within_int32(buffer, node)
                logger.warning(
                    f"Casting buffer {node.name} from torch.int64 to torch.int32"
                    f" defined in {node.meta.get('stack_trace','[no stack trace found]')}"
                )
                buffer_int32 = buffer.to(torch.int32)
                self.exported_program.state_dict[buffer_name] = buffer_int32
                continue

    def call(self, graph_module: torch.fx.GraphModule):
        self._to_int32(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
