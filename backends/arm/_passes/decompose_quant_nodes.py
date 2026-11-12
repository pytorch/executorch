# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.decompose_round_pass import DecomposeRoundPass
from executorch.backends.arm.constants import (
    DEQUANT_PER_TENSOR_OP,
    DEQUANT_PER_TENSOR_OP_T,
    QUANT_PER_TENSOR_OP,
    QUANT_PER_TENSOR_OP_T,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import Node


class DecomposeQuantNodesPass(ArmPass):
    """Decompose scalar and tensor per-tensor Q/DQ into primitive edge ops."""

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeRoundPass}

    _QUANT_OPS = (QUANT_PER_TENSOR_OP, QUANT_PER_TENSOR_OP_T)
    _DEQUANT_OPS = (DEQUANT_PER_TENSOR_OP, DEQUANT_PER_TENSOR_OP_T)

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in (
                *self._QUANT_OPS,
                *self._DEQUANT_OPS,
            ):
                continue

            # Preserve an explicit Q/DQ boundary when it is immediately
            # requantized; the existing pass relies on this behavior.
            if node.target in self._DEQUANT_OPS and all(
                user.target in self._QUANT_OPS for user in node.users
            ):
                continue
            if (
                node.target in self._QUANT_OPS
                and node.all_input_nodes
                and node.all_input_nodes[0].target in self._DEQUANT_OPS
            ):
                continue

            modified = True
            x, scale, zero_point, qmin, qmax, dtype = node.args[:6]
            is_quant = node.target in self._QUANT_OPS
            is_tensor_qparams = node.target in (
                QUANT_PER_TENSOR_OP_T,
                DEQUANT_PER_TENSOR_OP_T,
            )
            input_rank = x.meta["val"].ndim
            input_dtype = x.meta["val"].dtype
            output_dtype = node.meta["val"].dtype
            fp_dtype = output_dtype if not is_quant else input_dtype

            with graph_module.graph.inserting_before(node):
                if is_tensor_qparams:
                    if not isinstance(scale, Node) or not isinstance(zero_point, Node):
                        raise RuntimeError(
                            f"Tensor Q/DQ overload {node.target} expected tensor scale/zp nodes"
                        )
                    scale_fp = create_node(
                        graph_module.graph,
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        args=(scale,),
                        kwargs={"dtype": fp_dtype},
                        from_node=node,
                    )
                    if is_quant:
                        zp_value = create_node(
                            graph_module.graph,
                            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                            args=(zero_point,),
                            kwargs={"dtype": fp_dtype},
                            from_node=node,
                        )
                        scale_value = create_node(
                            graph_module.graph,
                            exir_ops.edge.aten.reciprocal.default,
                            args=(scale_fp,),
                            from_node=node,
                        )
                    else:
                        zp_value = create_node(
                            graph_module.graph,
                            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                            args=(zero_point,),
                            kwargs={"dtype": torch.int32},
                            from_node=node,
                        )
                        scale_value = scale_fp
                else:
                    scalar_scale = cast(float, scale)
                    scalar_scale = scalar_scale if not is_quant else 1.0 / scalar_scale
                    scale_value = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.full.default,
                        args=((1,) * input_rank, scalar_scale),
                        kwargs={"dtype": fp_dtype},
                    )
                    zp_value = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.full.default,
                        args=((1,) * input_rank, zero_point),
                        kwargs={"dtype": fp_dtype if is_quant else torch.int32},
                    )

                if is_quant:
                    scaled = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.mul.Tensor,
                        args=(x, scale_value),
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
                        args=(rounded, zp_value),
                        from_node=node,
                    )
                    clamped = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.clamp.default,
                        args=(shifted, float(qmin), float(qmax)),
                        from_node=node,
                    )
                    output = create_node(
                        graph_module.graph,
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        args=(clamped,),
                        kwargs={"dtype": dtype},
                        from_node=node,
                    )
                else:
                    x_i32 = create_node(
                        graph_module.graph,
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        args=(x,),
                        kwargs={"dtype": torch.int32},
                        from_node=node,
                    )
                    shifted = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.sub.Tensor,
                        args=(x_i32, zp_value),
                        from_node=node,
                    )
                    shifted_fp = create_node(
                        graph_module.graph,
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        args=(shifted,),
                        kwargs={"dtype": fp_dtype},
                        from_node=node,
                    )
                    output = create_node(
                        graph_module.graph,
                        exir_ops.edge.aten.mul.Tensor,
                        args=(shifted_fp, scale_value),
                        from_node=node,
                    )

                node.replace_all_uses_with(output)
                graph_module.graph.erase_node(node)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified=modified)
