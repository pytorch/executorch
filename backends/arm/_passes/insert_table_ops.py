# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from itertools import chain
from typing import Callable, cast, Dict, Iterator, Set

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa_quant_utils import QuantArgs
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
    in_dtype = args[0].dtype
    if in_dtype == torch.int8:
        return args[0]
    return args[0].to(dtype=torch.int32)


class TableOps:
    """
    Helper class for finding the corresponding table operator for a given Node.
    """

    # Targets that follow a straigtforward one-to-one mapping to their table op
    unary_table_ops: Dict[EdgeOpOverload, Callable[[torch.Tensor], torch.Tensor]] = {
        exir_ops.edge.aten.ceil.default: torch.ceil,
        exir_ops.edge.aten.erf.default: torch.erf,
        exir_ops.edge.aten.exp.default: torch.exp,
        exir_ops.edge.aten.floor.default: torch.floor,
        exir_ops.edge.aten.log.default: torch.log,
        exir_ops.edge.aten.reciprocal.default: torch.reciprocal,
        exir_ops.edge.aten.rsqrt.default: torch.rsqrt,
        exir_ops.edge.aten.sigmoid.default: torch.sigmoid,
        exir_ops.edge.aten.cos.default: torch.cos,
        exir_ops.edge.aten.sin.default: torch.sin,
        exir_ops.edge.aten.tanh.default: torch.tanh,
        exir_ops.edge.aten.hardsigmoid.default: torch.nn.functional.hardsigmoid,
        exir_ops.edge.aten.hardswish.default: torch.nn.functional.hardswish,
    }

    # Targets that must be treated explicitly
    special_table_ops: Set[EdgeOpOverload] = {
        exir_ops.edge.aten.pow.Tensor_Scalar,
        exir_ops.edge.aten.gelu.default,
    }

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program

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
                case exir_ops.edge.aten.pow.Tensor_Scalar:
                    # Exponent is a constant. Embed it into a lambda.
                    exp = cast(int, node.args[1])
                    return lambda x: torch.pow(x, exp).flatten()
                case exir_ops.edge.aten.gelu.default:
                    # If kwargs not present it is default "none"
                    approximate = cast(
                        str,
                        (
                            node.kwargs["approximate"]
                            if "approximate" in node.kwargs
                            else "none"
                        ),
                    )
                    return lambda x: torch.nn.functional.gelu(
                        x, approximate=approximate
                    ).flatten()
                case _:
                    # Op must be handled if it's inside self.special_ops
                    raise AssertionError("Unhandled table operation")
        else:
            raise KeyError("Table op for {target} does not exist")

    @staticmethod
    def included_ops() -> Iterator[EdgeOpOverload]:
        return chain(TableOps.unary_table_ops, TableOps.special_table_ops)


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

    def generate_8bit_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        """Compute LUT values for a INT8 TOSA.TABLE. Also returns 0 since no shifting is required after 8bit table.
        The INT8 table is a simple 256 value 1-1 LUT.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            x = in_quantargs.dequantize_value(x)
            x = torch_op(x)
            return out_quantargs.quantize_value(x)

        return (
            f(
                torch.linspace(
                    start=in_quantargs.qmin,
                    end=in_quantargs.qmax,
                    steps=256,
                    # use torch.int64 to avoid overflow when dequantizing (subtracting zp).
                    # e.g. torch.tensor(-50, dtype=torch.int8) - 100 == torch.tensor(106, dtype=torch.int8)
                    dtype=torch.int64,
                )
            ).to(dtype=torch.int8),
            0,
        )

    def generate_16_bit_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        """Compute LUT values for a INT16 TOSA.TABLE with 32 bit output.
        In practice the output is 23 bits that should be interpreted as 16 'whole' bits and 7 fractional bits, see
        the specification: https://www.mlplatform.org/tosa/tosa_spec.html#_table. This means that the output
        will interpreted as 2**7=128 times too large unless accounted for by rescaling down the table output.

        Quantization can be either int16 or int32 which means that the op output could be larger than the 23 bits from
        the TOSA.TABLE output. In that case, we need to rescale up the output.

        To handle this we need to:
        1) Make sure that our table values fit within 16 bits.
        2) Insert a rescale after the table to handle the x128 from the fractional bits and match the quantization.

        The function returns rescale_lshift which says how much to rescale after the table. This value can negative.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            # Dont use the 7 LSBs.
            x = in_quantargs.dequantize_value((x & ~0x7F))
            x = torch_op(x)
            return out_quantargs.quantize_value(x)

        lut_values = f(
            torch.linspace(
                start=in_quantargs.qmin,
                end=in_quantargs.qmax + 1,
                steps=513,
                # use torch.int64 to avoid overflow when dequantizing (subtracting zp).
                # e.g. torch.tensor(-50, dtype=torch.int8) - 100 == torch.tensor(106, dtype=torch.int8)
                dtype=torch.int64,
            )
        )
        # Calculate how much we need to shift table values to fit in 16 signed bits
        # ceil(log2(max absolute table value)) + 1 bit for signedness - 16
        # Example:
        #       Max value in the table is 70 000. We want to fit it in 16 signed bits.
        #       70 000=0b10001000101110000 (17 digits) has ceil(log2(70 000)) = ceil(16.095) = 17 bits.
        #       If we shift it 17-16=1 bit, we do get 16 bits (0b1000100010111000),
        #       but due to signedness this is a negative number! So we need to shift it one more bit.
        # Note: for out_quantargs.dtype=torch.int16, rshift == 0 and rescale_lshift = -7.
        rshift = int(torch.ceil(torch.log2(lut_values.abs().max()))) + 1 - 16
        # The 7 fractional bits are equivalent to a lshift of 7, so subtract 7 from the lshift we do.
        rescale_lshift = rshift - 7
        lut_values = lut_values >> rshift
        return lut_values.to(dtype=torch.int16), rescale_lshift

    def generate_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        match out_quantargs.dtype:
            case torch.int8:
                return self.generate_8bit_table_values(
                    torch_op, in_quantargs, out_quantargs
                )
            case torch.int16 | torch.int32:
                return self.generate_16_bit_table_values(
                    torch_op, in_quantargs, out_quantargs
                )
            case _:
                raise ValueError(
                    f"Unsupported output dtype for table: {out_quantargs.dtype}"
                )

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
                output_node = table_node
                # Expect exactly one quantization parameter for input and output
                if len(input_qparams) != 1:
                    raise ValueError(
                        f"InsertTableOpsPass expected exactly one input quantization parameter, "
                        f"got {len(input_qparams)} for node {node.name}"
                    )
                if len(output_qparams) != 1:
                    raise ValueError(
                        f"InsertTableOpsPass expected exactly one output quantization parameter, "
                        f"got {len(output_qparams)} for node {node.name}"
                    )

                # Generate table buffer and how much to lshift the table output.
                buffer, lshift = self.generate_table_values(
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

                if lshift != 0:
                    scale = 2.0**lshift
                    rescale_node = create_node(
                        graph=graph_module.graph,
                        op_target=torch.ops.tosa._rescale.default,
                        args=(table_node, output_qparams[0].dtype, scale, 0, 0),
                    )
                    output_node = rescale_node

                node.replace_all_uses_with(output_node)

            graph_module.graph.erase_node(node)

            output_node.meta["input_qparams"] = input_qparams
            output_node.meta["output_qparams"] = output_qparams
            modified = True

        if modified:
            # retrace the graph to update the fake tensor types
            graph_module = super().call(graph_module).graph_module

            graph_module.recompile()
        return PassResult(graph_module, modified)
