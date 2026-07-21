# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node, set_node_arg
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata


def _set_fake_tensor_meta(node: torch.fx.Node, value: torch.Tensor) -> None:
    node.meta["val"] = value
    node.meta["tensor_meta"] = _extract_tensor_metadata(value)


def _get_per_tensor_dequant_args(grid: torch.fx.Node, grid_qparams: QuantArgs) -> tuple:
    return (
        grid,
        grid_qparams.scale,
        grid_qparams.zp,
        grid_qparams.qmin,
        grid_qparams.qmax,
        grid_qparams.dtype,
    )


class InsertGridSamplerGridDequantPass(ArmPass):
    """Insert an explicit float boundary for quantized grid_sample grid inputs.

    This runs before quant-node decomposition so the standard Arm quant passes
    can legalize the inserted dequant op, and later VGF custom-op rewriting sees
    the expected float grid contract. Per-channel grid qparams are rejected
    because the follow-up decompose pass only legalizes per-tensor
    quantized_decomposed ops.

    """

    _passes_required_after: set[type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in list(graph_module.graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.grid_sampler_2d.default
            ):
                continue

            grid = node.args[1]
            if not isinstance(grid, torch.fx.Node):
                continue
            if grid.meta["val"].dtype.is_floating_point:
                continue

            grid_qparams = get_input_qparams(node).get(1)
            if grid_qparams is None:
                raise RuntimeError(
                    "Quantized grid_sampler grid input is missing input qparams"
                )
            if grid_qparams.per_channel:
                raise RuntimeError(
                    "grid_sampler grid dequant only supports per-tensor qparams"
                )

            with graph_module.graph.inserting_before(node):
                dequant_grid = create_node(
                    graph_module.graph,
                    op_target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                    args=_get_per_tensor_dequant_args(grid, grid_qparams),
                    from_node=node,
                )
                _set_fake_tensor_meta(
                    dequant_grid, grid_qparams.dequantize_value(grid.meta["val"])
                )

            set_node_arg(node, 1, dequant_grid)
            if "input_qparams" in node.meta:
                node.meta["input_qparams"] = {
                    idx: qargs
                    for idx, qargs in node.meta["input_qparams"].items()
                    if idx != 1
                }
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)
