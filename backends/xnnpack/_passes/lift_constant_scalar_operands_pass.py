# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from numbers import Number
from typing import Dict, Optional, Union

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch._ops import OpOverload


ScalarOp = Union[EdgeOpOverload, OpOverload]


class LiftConstantScalarOperandsPass(ExportPass):
    """
    Lift scalar operands into tensor constants for selected binary ops.

    XNNPACK already supports the tensor overloads for these binary operations.
    This pass converts explicitly listed scalar overloads to their tensor
    overloads by replacing constant scalar operands with small tensor constants.
    The constants are registered as buffers so they do not become portable
    ``full`` kernels. Keep the op map narrow until each new scalar overload is
    covered by tests.
    """

    default_scalar_to_tensor_ops: Dict[ScalarOp, ScalarOp] = {
        exir_ops.edge.aten.mul.Scalar: exir_ops.edge.aten.mul.Tensor,
    }
    sdpa_passthrough_ops = {
        exir_ops.edge.aten.expand_copy.default,
        exir_ops.edge.aten.view_copy.default,
    }

    def __init__(
        self,
        scalar_to_tensor_ops: Optional[Dict[ScalarOp, ScalarOp]] = None,
    ) -> None:
        super().__init__()
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
        name = self._get_new_attr_name(graph_module)
        # Keep constants as module attributes so the portable path can emit them
        # without introducing aten.full, while XNNPACK can still read them as params.
        graph_module.register_buffer(name, tensor)

        fake_mode = node.meta["val"].fake_mode
        with graph_module.graph.inserting_before(node):
            constant_node = graph_module.graph.get_attr(name)
            constant_node.meta["val"] = fake_mode.from_tensor(
                tensor, static_shapes=True
            )
        return constant_node

    def _get_new_attr_name(self, graph_module: torch.fx.GraphModule) -> str:
        prefix = "_tensor_constant_"
        index = 0
        while hasattr(graph_module, f"{prefix}{index}"):
            index += 1
        return f"{prefix}{index}"

    def _feeds_sdpa_qk_bmm(self, node: torch.fx.Node) -> bool:
        """
        Return true for the scale muls consumed by XNNPACK's SDPA pattern.

        ConvertToSDPAPass recovers the user-specified attention scale from the
        pre-QK^T ``aten.mul.Scalar`` nodes. Keep those scalar muls intact so
        SDPA conversion can still find the scale before replacing the pattern.
        """
        users_to_visit = list(node.users)
        visited = set()
        while users_to_visit:
            user = users_to_visit.pop()
            if user in visited:
                continue
            visited.add(user)

            if (
                user.op == "call_function"
                and user.target == exir_ops.edge.aten.bmm.default
            ):
                return True

            if user.op == "call_function" and user.target in self.sdpa_passthrough_ops:
                users_to_visit.extend(user.users)

        return False

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

            if (
                node.target == exir_ops.edge.aten.mul.Scalar
                and self._feeds_sdpa_qk_bmm(node)
            ):
                continue

            input_value = node.args[0].meta.get("val")
            output_value = node.meta.get("val")
            if (
                input_value is None
                or output_value is None
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
