# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable, cast, Dict, Set

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa_quant_utils import QuantArgs
from executorch.backends.transforms.utils import delete_constant_placeholder
from executorch.exir import ExportedProgram

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule
from torch.fx.node import Node
from torch.library import impl, Library

lib = Library("tosa", "DEF")
lib.define("_table(Tensor self) -> Tensor")


@impl(lib, "_table")
def _table_impl(*args, **kwargs):  # pyre-ignore
    return args[0]


class TableOps:
    """
    Helper class for finding the corresponding table operator for a given Node.
    """

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program

        # Targets that follow a straigtforward one-to-one mapping to their table op
        self.unary_table_ops: Dict[
            EdgeOpOverload, Callable[[torch.Tensor], torch.Tensor]
        ] = {
            exir_ops.edge.aten.exp.default: torch.exp,
            exir_ops.edge.aten.floor.default: torch.floor,
            exir_ops.edge.aten.log.default: torch.log,
            exir_ops.edge.aten.reciprocal.default: torch.reciprocal,
            exir_ops.edge.aten.rsqrt.default: torch.rsqrt,
            exir_ops.edge.aten.sigmoid.default: torch.sigmoid,
            exir_ops.edge.aten.tanh.default: torch.tanh,
            exir_ops.edge.aten.hardsigmoid.default: torch.nn.functional.hardsigmoid,
            exir_ops.edge.aten.hardswish.default: torch.nn.functional.hardswish,
        }

        # Targets that must be treated explicitly
        self.special_table_ops: Set[EdgeOpOverload] = {
            exir_ops.edge.aten.pow.Tensor_Tensor,
        }

    def __contains__(self, node: Node) -> bool:
        return (
            node.target in self.unary_table_ops or node.target in self.special_table_ops
        )

    def __getitem__(self, node: Node):
        target = cast(EdgeOpOverload, node.target)
        if target in self.unary_table_ops:
            return self.unary_table_ops[target]
        elif target in self.special_table_ops:
            match target:
                case exir_ops.edge.aten.pow.Tensor_Tensor:
                    # Exponent is a constant. Retrieve it from the graph and embed it into a lambda.
                    exp_node = cast(Node, node.args[1])
                    exp_name = self.exported_program.graph_signature.inputs_to_buffers[
                        exp_node.name
                    ]
                    exp = self.exported_program.state_dict[exp_name]
                    return lambda x: torch.pow(x, exp).flatten()
                case _:
                    raise NotImplementedError("Unhandled table operation")
        else:
            raise KeyError("Table op for {target} does not exist")


class InsertTableOpsPass(ExportPass):
    """
    For ops in self.table_ops they need to be serialized as a TOSA TABLE. This pass replaces these
    edge ops with a tosa._table(input: Tensor, target_str: str) where target_str == str(node.target).
    When lowering the _table node target_str will be used to find the corresponding torch operator
    which will be used to produce the table values in operators/op_table.py.
    """

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.table_ops = TableOps(exported_program)

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
            if node.op != "call_function" or node not in self.table_ops:
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
                    op_target=torch.ops.tosa._table.default,
                    args=(node.args[0],),
                )
                assert len(input_qparams) == 1
                assert len(output_qparams) == 1
                # Generate table buffer
                buffer = self.generate_table_values(
                    torch_op=self.table_ops[node],
                    in_quantargs=input_qparams[0],
                    out_quantargs=output_qparams[0],
                )
                # Register buffer in self.exported_program.state_dict
                # When the graph is retraced, the implementation _table is used and the suffix _default disappears from the node name
                # Remove it here to make it possible to find in the node_visitor
                self.register_buffer(
                    buffer_name=table_node.name.replace("_default", ""), buffer=buffer
                )
                node.replace_all_uses_with(table_node)

            if node.target in self.table_ops.special_table_ops:
                # The node must be treated explicitly
                match node.target:
                    case exir_ops.edge.aten.pow.Tensor_Tensor:
                        exp_node = node.args[1]
                        graph_module.graph.erase_node(node)
                        delete_constant_placeholder(self.exported_program, exp_node)
                    case _:
                        raise NotImplementedError("Unhandled table operation")
            else:
                graph_module.graph.erase_node(node)

            table_node.meta["input_qparams"] = input_qparams
            table_node.meta["output_qparams"] = output_qparams
            modified = True

        if modified:
            # retrace the graph to update the fake tensor types
            graph_module = super().call(graph_module).graph_module

            graph_module.recompile()
        return PassResult(graph_module, modified)
