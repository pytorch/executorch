# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node, set_node_arg
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
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


def _can_keep_quantized_grid(grid_qparams: QuantArgs) -> bool:
    return not grid_qparams.per_channel and grid_qparams.dtype == torch.int8


def _uses_grid_sampler_int8_snorm_qparams(qparams: QuantArgs) -> bool:
    return (
        not qparams.per_channel
        and math.isclose(
            qparams.get_scale_per_tensor(), 1.0 / 127.0, rel_tol=1e-6, abs_tol=1e-9
        )
        and qparams.get_zp_per_tensor() == 0
        and qparams.qmin == -127
        and qparams.qmax == 127
        and qparams.dtype == torch.int8
    )


def _supports_quantized_grid_sampler_path(node: torch.fx.Node) -> bool:
    try:
        input_qparams = get_input_qparams(node)
        output_qparams = get_output_qparams(node)
    except ValueError:
        return False

    image_qparams = input_qparams.get(0)
    if image_qparams is None or not output_qparams:
        return False
    if not _uses_grid_sampler_int8_snorm_qparams(image_qparams):
        return False
    if not _uses_grid_sampler_int8_snorm_qparams(next(iter(output_qparams.values()))):
        return False

    input_tensor = node.args[0]
    interpolation_mode = node.args[2]
    if not isinstance(input_tensor, torch.fx.Node):
        return False
    if not isinstance(interpolation_mode, int):
        return False
    input_val = input_tensor.meta.get("val")
    if not isinstance(input_val, torch.Tensor) or len(input_val.shape) != 4:
        return False
    if int(input_val.shape[0]) != 1 or interpolation_mode not in (0, 1):
        return False
    return int(input_val.shape[1]) in (3, 4)


class InsertGridSamplerGridDequantPass(ArmPass):
    """Insert an explicit float boundary for quantized grid_sample grid inputs.

    This runs before quant-node decomposition so the standard Arm quant passes
    can legalize the inserted dequant op. For supported per-tensor int8 grids we
    preserve the quantized grid through to the VGF custom-op rewrite so the
    shader can dequantize coordinates internally. Unsupported per-tensor qparams
    are still dequantized to float here, but that does not create a mixed
    int8-image / float-grid shader mode; the supported shader modes remain
    float/float and int8/int8. Per-channel grid qparams are rejected because the
    follow-up decompose pass only legalizes per-tensor quantized_decomposed ops.

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
            if _can_keep_quantized_grid(
                grid_qparams
            ) and _supports_quantized_grid_sampler_path(node):
                continue
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
