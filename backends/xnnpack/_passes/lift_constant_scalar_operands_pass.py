# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from numbers import Number
from typing import Dict, Optional, Union

import torch
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult
from torch._ops import OpOverload
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

ScalarOp = Union[EdgeOpOverload, OpOverload]


class LiftConstantScalarOperandsPass(XNNPACKPass):
    """
    Lift scalar operands into tensor constants for selected binary ops.

    XNNPACK already supports the tensor overloads for these binary operations.
    This pass converts explicitly listed scalar overloads to their tensor
    overloads by replacing constant scalar operands with small tensor constants.
    The constants are registered as exported-program constant tensor inputs.
    Keep the op map narrow until each new scalar overload is covered by tests.
    """

    default_scalar_to_tensor_ops: Dict[ScalarOp, ScalarOp] = {
        exir_ops.edge.aten.mul.Scalar: exir_ops.edge.aten.mul.Tensor,
    }

    def __init__(
        self,
        exported_program: ExportedProgram,
        scalar_to_tensor_ops: Optional[Dict[ScalarOp, ScalarOp]] = None,
    ) -> None:
        super().__init__(exported_program)
        self.scalar_to_tensor_ops = (
            scalar_to_tensor_ops
            if scalar_to_tensor_ops is not None
            else self.default_scalar_to_tensor_ops
        )

    def _create_constant_node(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        value: Number,
    ) -> torch.fx.Node:
        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            raise RuntimeError("Expected scalar op input to be an FX node.")

        input_value = input_node.meta["val"]
        tensor = torch.tensor(value, dtype=input_value.dtype, device=input_value.device)
        name = self._get_new_constant_name(graph_module)
        first_placeholder = next(
            graph_node
            for graph_node in graph_module.graph.nodes
            if graph_node.op == "placeholder"
        )
        with graph_module.graph.inserting_before(first_placeholder):
            return create_constant_placeholder(
                self.exported_program,
                graph_module.graph,
                name,
                InputKind.CONSTANT_TENSOR,
                tensor,
            )

    def _get_new_constant_name(self, graph_module: torch.fx.GraphModule) -> str:
        prefix = "_tensor_constant_"
        existing_names = {node.name for node in graph_module.graph.nodes}
        existing_names.update(self.exported_program.constants)
        existing_names.update(self.exported_program.state_dict)
        index = 0
        while f"{prefix}{index}" in existing_names:
            index += 1
        return f"{prefix}{index}"

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        for node in list(graph_module.graph.nodes):
            if (
                node.op != "call_function"
                or node.target not in self.scalar_to_tensor_ops
                or len(node.args) != 2
                or not isinstance(node.args[0], torch.fx.Node)
                or not isinstance(node.args[1], Number)
            ):
                continue

            input_value = node.args[0].meta.get("val")
            output_value = node.meta.get("val")
            if (
                not isinstance(input_value, torch.Tensor)
                or not isinstance(output_value, torch.Tensor)
                or input_value.dtype != output_value.dtype
            ):
                continue

            tensor_arg = self._create_constant_node(graph_module, node, node.args[1])
            node.args = (node.args[0], tensor_arg)
            node.target = self.scalar_to_tensor_ops[node.target]
            modified = True

        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()

        return PassResult(graph_module, modified)
