# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.samsung.utils.constants import QuantConstants
from executorch.backends.transforms.utils import get_param_tensor, is_param_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export import ExportedProgram


class AnnotateScalarParametersPass(ExportPass):
    """
    Need to add quantization parameters for scalars for some ops
    Ifm(Quantized)------TargetOP---
    Scalar(Non-Quant)---/
    Notice: Such scalars are converted to tensor node by default pass
    """

    TARGET_OPS = {
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.pow.Tensor_Tensor,
        exir_ops.edge.aten.where.self,
    }

    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def _is_param(self, node: torch.fx.Node) -> bool:
        return node.op in ("placeholder", "get_attr") and is_param_node(
            self.edge_program, node
        )

    def annotate(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target not in self.TARGET_OPS or "quantize_attrs" not in node.meta:
                continue
            if node.target == exir_ops.edge.aten.where.self:
                # where.self(condition, self, other): the condition is a bool
                # mask, so the scalar sits in one of the two value inputs.
                input0, input1 = node.args[1], node.args[2]
            else:
                input0, input1 = node.all_input_nodes[0], node.all_input_nodes[1]
            if not self._is_param(input0):
                if not self._is_param(input1):
                    continue
                ifm_node, param_tensor_node = input0, input1
            else:
                ifm_node, param_tensor_node = input1, input0
            if not (quantize_attrs := ifm_node.meta.get("quantize_attrs")):
                continue
            param_tensor = get_param_tensor(self.edge_program, param_tensor_node)
            if not param_tensor.shape:
                if param_tensor_node.meta.get("quantize_attrs"):
                    # Already has complete qparams of its own, dtype range included.
                    continue
                zero_point = 0
                scale = abs(float(param_tensor))
            else:
                continue
            q_dtype = quantize_attrs[QuantConstants.QUANT_KEY.quant_dtype]
            if scale == 0:
                scale = 1.0
            qparams = {
                QuantConstants.QUANT_KEY.scale: scale,
                QuantConstants.QUANT_KEY.quant_dtype: q_dtype,
                QuantConstants.QUANT_KEY.quant_max: torch.iinfo(q_dtype).max,
                QuantConstants.QUANT_KEY.quant_min: torch.iinfo(q_dtype).min,
                QuantConstants.QUANT_KEY.zero_point: zero_point,
            }
            param_tensor_node.meta["quantize_attrs"] = qparams

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        self.annotate(graph_module)
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
