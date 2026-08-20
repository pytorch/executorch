# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.samsung.quantizer.quantizer import global_quant_info
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
    }

    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def annotate(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target not in self.TARGET_OPS or "quantize_attrs" not in node.meta:
                continue
            torch_quant_dtype = global_quant_info.weight_precison.torch_dtype
            for input_arg in node.all_input_nodes:
                if input_arg.op not in ("placeholder", "get_attr") or not is_param_node(
                    self.edge_program, input_arg
                ):
                    continue
                else:
                    tensor = get_param_tensor(self.edge_program, input_arg)
                    if not tensor.shape:
                        qparams = {
                            QuantConstants.QUANT_KEY.scale: float(tensor),
                            QuantConstants.QUANT_KEY.quant_dtype: torch_quant_dtype,
                            QuantConstants.QUANT_KEY.quant_max: torch.iinfo(
                                torch_quant_dtype
                            ).max,
                            QuantConstants.QUANT_KEY.quant_min: torch.iinfo(
                                torch_quant_dtype
                            ).min,
                            QuantConstants.QUANT_KEY.zero_point: 0,
                        }
                        input_arg.meta["quantize_attrs"] = qparams

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        self.annotate(graph_module)
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
