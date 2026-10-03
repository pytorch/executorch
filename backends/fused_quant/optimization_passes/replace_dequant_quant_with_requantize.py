# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.graph_utils import add_constant
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch import fx
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


def _lift_qparams(
    exported_program: ExportedProgram,
    node: fx.Node,
    name: str,
) -> tuple[fx.Node, fx.Node, torch.dtype, int, int]:
    scale = add_constant(
        exported_program,
        f"_{name}_scale",
        torch.tensor(get_arg(node, "scale", float), dtype=torch.float32),
        node,
        InputKind.CONSTANT_TENSOR,
    )
    zero_point = add_constant(
        exported_program,
        f"_{name}_zero_point",
        torch.tensor(get_arg(node, "zero_point", int), dtype=torch.int64),
        node,
        InputKind.CONSTANT_TENSOR,
    )
    return (
        scale,
        zero_point,
        get_arg(node, "dtype", torch.dtype),
        get_arg(node, "quant_min", int),
        get_arg(node, "quant_max", int),
    )


class ReplaceDequantQuantWithRequantize(ExportedProgramPassBase):
    """Replace direct per-tensor dequantize-quantize branches with requantize."""

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False

        dequant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        for dequant in dequant_nodes:
            quant_nodes = [
                user
                for user in dequant.users
                if user.target
                == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            ]
            if not quant_nodes:
                continue

            inp_scale, inp_zero_point, _, inp_quant_min, inp_quant_max = _lift_qparams(
                exported_program, dequant, "requantize_inp"
            )
            inp_qparams = (
                inp_scale,
                inp_zero_point,
                dequant.meta["val"].dtype,
                inp_quant_min,
                inp_quant_max,
            )

            for quant in quant_nodes:
                out_qparams = _lift_qparams(exported_program, quant, "requantize_out")
                with graph.inserting_before(quant):
                    requantize = graph.call_function(
                        exir_ops.edge.fused_quant.requantize.default,
                        args=(dequant.args[0], *inp_qparams, *out_qparams),
                    )
                    requantize.meta = quant.meta.copy()

                quant.replace_all_uses_with(requantize)
                graph.erase_node(quant)
            if not dequant.users:
                graph.erase_node(dequant)
            modified = True

        if modified:
            exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, modified)
