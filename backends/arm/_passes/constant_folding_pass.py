# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.passes.constant_folding import constant_fold


class ConstantFoldingPass(ArmPass):
    """Fold constant subgraphs using torch's export constant folding pass. To be used before to_edge transform."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    parameter_targets = {
        torch.ops.aten.linear.default,
        torch.ops.aten.convolution.default,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv1d.padding,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv2d.padding,
        torch.ops.aten.conv3d.default,
        torch.ops.aten.conv3d.padding,
        torch.ops.aten.conv_transpose2d.input,
    }

    def _ensure_param_attr(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node | None
    ) -> bool:
        """
        Replaces tensor attributes with parameter attributes.
        """
        if node is None or node.op != "get_attr":
            return False
        target = node.target
        try:
            attr = getattr(graph_module, target)  # type: ignore[arg-type]
        except AttributeError:
            return False
        if isinstance(attr, torch.nn.Parameter):
            return False
        if not isinstance(attr, torch.Tensor):
            return False
        setattr(graph_module, target, torch.nn.Parameter(attr, requires_grad=False))  # type: ignore[arg-type]
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        before_graph = str(graph_module.graph)

        # The ConstantFolder checks for this attribute which does not always exist
        faked_dequantize_affine = False
        if not hasattr(torch.ops.pt2e_quant, "dequantize_affine"):
            torch.ops.pt2e_quant.dequantize_affine = None  # type: ignore[attr-defined]
            faked_dequantize_affine = True

        constant_fold(graph_module)

        # Remove attribute again if added to not affect global scope
        if faked_dequantize_affine:
            del torch.ops.pt2e_quant.dequantize_affine  # type: ignore[attr-defined]

        modified = before_graph != str(graph_module.graph)

        # Constant folding may have replaced some parameters with tensors, so we need to ensure they are still parameters for the backend to work correctly.
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in self.parameter_targets:
                args = node.args
                if len(args) > 1:
                    modified |= self._ensure_param_attr(graph_module, args[1])
                if len(args) > 2:
                    modified |= self._ensure_param_attr(graph_module, args[2])

        return PassResult(graph_module, modified)
