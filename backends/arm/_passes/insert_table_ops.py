# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa_quant_utils import QuantArgs
from executorch.exir import ExportedProgram

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule
from torch.library import impl, Library

lib = Library("tosa", "DEF")
lib.define("_table(Tensor self) -> Tensor")


@impl(lib, "_table")
def _table_impl(*args, **kwargs):  # pyre-ignore
    return args[0]


class InsertTableOpsPass(ExportPass):
    """
    For ops in self.table_ops they need to be serialized as a TOSA TABLE. This pass replaces these
    edge ops with a tosa._table(input: Tensor, target_str: str) where target_str == str(node.target).
    When loweringthe _table node target_str will be used to find the corresponding torch operator
    which will be used to produce the table values in operators/op_table.py.
    """

    table_ops: Dict[EdgeOpOverload, Callable[[torch.Tensor], torch.Tensor]] = {
        exir_ops.edge.aten.exp.default: torch.exp,
        exir_ops.edge.aten.log.default: torch.log,
        exir_ops.edge.aten.reciprocal.default: torch.reciprocal,
        exir_ops.edge.aten.rsqrt.default: torch.rsqrt,
        exir_ops.edge.aten.sigmoid.default: torch.sigmoid,
        exir_ops.edge.aten.tanh.default: torch.tanh,
    }

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    def register_buffer(self, buffer_name: str, buffer: torch.Tensor) -> None:
        """
        Add buffer to self.exported_program.state_dict
        """
        self.exported_program.state_dict[buffer_name] = buffer

    def generate_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> torch.Tensor:
        def f(x: torch.Tensor) -> torch.Tensor:
            x = in_quantargs.dequantize_value(x)
            x = torch_op(x)
            return out_quantargs.quantize_value(x)

        input_dtype = in_quantargs.dtype
        steps = in_quantargs.qmax - in_quantargs.qmin + 1
        return f(
            torch.linspace(
                start=in_quantargs.qmin,
                end=in_quantargs.qmax,
                steps=steps,
                # use torch.int64 to avoid overflow when dequantizing (subtracting zp).
                # e.g. torch.tensor(-50, dtype=torch.int8) - 100 == torch.tensor(106, dtype=torch.int8)
                dtype=torch.int64,
            )
        ).to(dtype=input_dtype)

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in self.table_ops:
                continue
            input_qparams = node.meta["input_qparams"]
            output_qparams = node.meta["output_qparams"]
            if len(input_qparams) == 0 or len(output_qparams) == 0:
                # We only want to replace the node if it's quantized
                continue
            # Create table node
            with graph_module.graph.inserting_before(node):
                table_node = create_node(
                    graph=graph_module.graph,
                    op_target=torch.ops.tosa._table,
                    args=(node.args[0],),
                )
                assert len(input_qparams) == 1
                assert len(output_qparams) == 1
                # Generate table buffer
                buffer = self.generate_table_values(
                    torch_op=self.table_ops[node.target],
                    in_quantargs=input_qparams[0],
                    out_quantargs=output_qparams[0],
                )
                # Register buffer in self.exported_program.state_dict
                self.register_buffer(buffer_name=table_node.name, buffer=buffer)
                node.replace_all_uses_with(table_node)
            graph_module.graph.erase_node(node)
            table_node.meta["input_qparams"] = input_qparams
            table_node.meta["output_qparams"] = output_qparams
            modified = True

        if modified:
            # retrace the graph to update the fake tensor types
            graph_module = super().call(graph_module).graph_module

            graph_module.recompile()
        return PassResult(graph_module, modified)
