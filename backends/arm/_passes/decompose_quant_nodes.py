# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.decompose_round_pass import DecomposeRoundPass
from executorch.backends.arm.constants import DEQUANT_PER_TENSOR_OP, QUANT_PER_TENSOR_OP
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeQuantNodesPass(ArmPass):
    """Decomposes quantization nodes into more primitive operations by rewriting the graph
    using the two formulas:

    quantized value = clamp(round(fp32_value / scale) + zero point, qmin, qmax)

    fp32_value = (quantized value - zp) * scale

    For quantization nodes, the pass replaces them with:

    1. Multiplying the input by the inverse of the scale factor.
    2. Rounding the result.
    3. Adding the zero point.
    4. Clamping the result to [qmin, qmax].
    5. Casting to the target data type.

    For dequantization nodes, the pass replaces them with:

    1. Casting the input to int32.
    2. Subtracting the zero point.
    3. Casting to float32.
    4. Multiplying by the scale factor.

    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeRoundPass}

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in (
                QUANT_PER_TENSOR_OP,
                DEQUANT_PER_TENSOR_OP,
            ):
                continue
            if node.target == DEQUANT_PER_TENSOR_OP and all(
                user.target == QUANT_PER_TENSOR_OP for user in node.users
            ):
                continue
            elif (
                node.target == QUANT_PER_TENSOR_OP
                and node.all_input_nodes[0].target == DEQUANT_PER_TENSOR_OP
            ):
                continue
            modified = True
            args = node.args
            input_rank = args[0].meta["val"].ndim
            x, scale, zero_point, qmin, qmax, dtype = args
            # Instead of dividing by scale in quantization, we multiply by 1/scale
            # when quantizing.
            scale = cast(float, scale)
            scale = scale if node.target == DEQUANT_PER_TENSOR_OP else 1.0 / scale
            with graph_module.graph.inserting_before(node):
                scale_const = create_node(
                    graph_module.graph,
                    exir_ops.edge.aten.full.default,
                    args=((1,) * input_rank, scale),
                    kwargs={"dtype": torch.float32},
                )
                zp_const = create_node(
                    graph_module.graph,
                    exir_ops.edge.aten.full.default,
                    args=((1,) * input_rank, zero_point),
                    kwargs={
                        "dtype": (
                            torch.float32
                            if node.target == QUANT_PER_TENSOR_OP
                            else torch.int32
                        )
                    },
                )
                if node.target == QUANT_PER_TENSOR_OP:
                    # TODO MLETORCH-1587: Decompose quantization nodes using more integer arithmetic
                    scaled = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.mul.Tensor,
                        args=(x, scale_const),
                        from_node=node,
                    )
                    rounded = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.round.default,
                        args=(scaled,),
                        from_node=node,
                    )
                    shifted = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.add.Tensor,
                        args=(rounded, zp_const),
                        from_node=node,
                    )
                    clamped = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.clamp.default,
                        args=(shifted, float(qmin), float(qmax)),
                        from_node=node,
                    )
                    quantized = create_node(
                        graph_module.graph,
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        args=(clamped,),
                        kwargs={"dtype": dtype},
                        from_node=node,
                    )
                    output = quantized
                else:
                    input_casted_to_zp_dtype = create_node(
                        graph_module.graph,
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        args=(x,),
                        kwargs={"dtype": torch.int32},
                        from_node=node,
                    )
                    shifted = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.sub.Tensor,
                        args=(input_casted_to_zp_dtype, zp_const),
                        from_node=node,
                    )
                    casted_to_float = create_node(
                        graph_module.graph,
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        args=(shifted,),
                        kwargs={"dtype": torch.float32},
                        from_node=node,
                    )
                    dequantized = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.mul.Tensor,
                        args=(casted_to_float, scale_const),
                        from_node=node,
                    )
                    output = dequantized
                node.replace_all_uses_with(output)
                graph_module.graph.erase_node(node)
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified=modified)
