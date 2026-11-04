# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.cast_int_comparison_inputs_pass import (
    CastIntComparisonInputsPass,
)
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import is_buffer
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)


class CastInt64BuffersToInt32Pass(ArmPass):
    """Cast int64 buffers to int32 if the int64 data is in int32 range.

    With the int64 extension under PRO-INT, a buffer whose users all compute in
    int64, compare it or return it keeps int64.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _comparison_ops = CastIntComparisonInputsPass.target_ops | {
        exir_ops.edge.aten.eq.Scalar,
        exir_ops.edge.aten.ne.Scalar,
        exir_ops.edge.aten.ge.Scalar,
        exir_ops.edge.aten.gt.Scalar,
        exir_ops.edge.aten.le.Scalar,
        exir_ops.edge.aten.lt.Scalar,
    }

    def __init__(self, exported_program: ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def should_run_pass(self, graph_module: torch.fx.GraphModule) -> bool:
        for node in graph_module.graph.nodes:
            if len(node.users) == 0:
                continue
            fake_tensor = node.meta.get("val")
            if not isinstance(fake_tensor, torch._subclasses.fake_tensor.FakeTensor):
                continue
            if fake_tensor.dtype == torch.int64 and is_buffer(
                self.exported_program, node
            ):
                return True
        return False

    def _assert_within_int32(self, tensor: torch.Tensor, node: torch.fx.Node):
        if torch.min(tensor) < torch.iinfo(torch.int32).min:
            raise RuntimeError(
                f"Node {node.name} has value < {torch.iinfo(torch.int32).min}"
            )
        if torch.max(tensor) > torch.iinfo(torch.int32).max:
            raise RuntimeError(
                f"Node {node.name} has value > {torch.iinfo(torch.int32).max}"
            )

    def _cast_buffer_to_int32(self, node: torch.fx.Node) -> bool:
        buffer_name = self.exported_program.graph_signature.inputs_to_buffers[node.name]
        if buffer_name in self.exported_program.state_dict:
            store = self.exported_program.state_dict
        elif buffer_name in self.exported_program.constants:
            # Non-persistent buffers are tracked by inputs_to_buffers, but their
            # tensor values live in ExportedProgram.constants instead of the
            # state_dict. Examples include transformer position_ids buffers.
            store = self.exported_program.constants
        else:
            logger.warning(
                "Skipping int64 buffer %s (%s): value not found in state_dict "
                "or constants",
                node.name,
                buffer_name,
            )
            return False

        buffer = store[buffer_name]
        self._assert_within_int32(buffer, node)
        store[buffer_name] = buffer.to(torch.int32)
        node.meta["val"] = node.meta["val"].to(torch.int32)
        return True

    @classmethod
    def _users_compute_in_int64(cls, node: torch.fx.Node) -> bool:
        def is_int64(value) -> bool:
            return isinstance(value, torch.Tensor) and value.dtype == torch.int64

        return all(
            user.op == "output"
            or user.target in cls._comparison_ops
            or is_int64(user.meta.get("val"))
            or any(
                is_int64(arg.meta.get("val"))
                for arg in user.all_input_nodes
                if arg is not node
            )
            for user in node.users
        )

    def _to_int32(self, graph_module: torch.fx.GraphModule) -> int:
        tosa_spec = get_context_spec()
        keep_int64 = tosa_spec.support_integer() and tosa_spec.support_extension(
            "int64"
        )
        converted = 0
        for node in graph_module.graph.nodes:
            if len(node.users) == 0:
                continue
            if "val" not in node.meta:
                continue
            fake_tensor = node.meta["val"]
            if not isinstance(fake_tensor, torch._subclasses.fake_tensor.FakeTensor):
                continue
            if fake_tensor.dtype != torch.int64:
                continue
            if keep_int64 and self._users_compute_in_int64(node):
                continue
            if is_buffer(self.exported_program, node):
                converted += self._cast_buffer_to_int32(node)
        return converted

    def call(self, graph_module: torch.fx.GraphModule):
        converted = self._to_int32(graph_module)
        modified = converted > 0
        if modified:
            graph_module = super().call(graph_module).graph_module
            logger.warning(
                "CastInt64BuffersToInt32Pass: cast %d int64 buffer(s) to int32.",
                converted,
            )
        return PassResult(graph_module, modified)
