# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.fused_quant.graph_utils import (
    compute_meta_val,
    get_constant,
    get_qparams_from_node,
    get_scale,
    get_zero_point,
    set_constant,
)
from executorch.backends.fused_quant.ops import QuantParamsStruct
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch.export import ExportedProgram


def _try_prequantize_embedding(ep: ExportedProgram, node: fx.Node) -> bool:
    """Fold a quantized-table embedding into a plain int8 gather.

    A fused_quant.embedding with a quantized table (weight qparams) AND a
    per-tensor quantized output is a pure constant transform of the table: each
    gathered row is dequantized (group/per-channel) and then requantized
    per-tensor. Because the gather commutes with the elementwise (de)quantize,
    we apply that transform to the whole table at compile time and replace the op
    with a plain aten.embedding over the resulting int8 table -- dropping the
    scale/zero_point constants and all runtime (de)quant work. The result is
    bit-exact with what the op would compute, and the gathered int8 values feed
    downstream ops directly (their input qparams already equal the former output
    qparams).

    Embeddings whose output stays float (no per-tensor output quant) are left for
    the cadence quantized_embedding_byte path.
    """
    weight_qp = get_qparams_from_node(node, "weight")
    out_qp = get_qparams_from_node(node, "out")
    if weight_qp is None or out_qp is None or not out_qp.is_per_tensor():
        return False

    weight_node = get_arg(node, "weight", fx.Node)
    indices = get_arg(node, "indices", fx.Node)

    table = get_constant(ep, weight_node)
    weight_scale = get_constant(ep, weight_qp.scale)
    weight_zero_point = get_constant(ep, weight_qp.zero_point)
    if table is None or weight_scale is None or weight_zero_point is None:
        return False

    # Per-tensor output qparams are 0-dim lifted constants.
    out_scale = get_scale(ep, out_qp)
    out_zero_point = get_zero_point(ep, out_qp)

    # Recompute the table exactly as the op would at runtime: dequantize the
    # (group/per-channel) table to float, then quantize it per-tensor.
    table_qp = QuantParamsStruct(
        scale=weight_scale,
        zero_point=weight_zero_point,
        dtype=weight_qp.dtype,
        quant_min=weight_qp.quant_min,
        quant_max=weight_qp.quant_max,
    )
    output_qp = QuantParamsStruct(
        scale=torch.tensor(out_scale),
        zero_point=torch.tensor(out_zero_point),
        dtype=out_qp.dtype,
        quant_min=out_qp.quant_min,
        quant_max=out_qp.quant_max,
    )
    prequantized = output_qp.quantize(table_qp.dequantize(table))
    # Same shape/dtype (int8 table) as the original, so the placeholder's meta
    # stays valid -- only the backing values change.
    set_constant(ep, weight_node, prequantized)

    graph = node.graph
    with graph.inserting_before(node):
        gather = graph.call_function(
            exir_ops.edge.aten.embedding.default,
            args=(weight_node, indices),
        )
        gather.meta = node.meta
        gather.meta["val"] = compute_meta_val(gather)
    node.replace_all_uses_with(gather)
    graph.erase_node(node)
    return True


class PrequantizeEmbedding(ExportedProgramPassBase):
    """Replace a quantized-table fused_quant.embedding with a plain int8 gather.

    Runs before coloring. For an embedding whose table is quantized and whose
    output is per-tensor quantized (after QuantAbsorptionPass folds the consumer's
    quantize into the op's output qparams), the table is pre-quantized at compile
    time and the op becomes aten.embedding over the resulting int8 table. The
    now-dead scale/zero_point constants are removed by constant_prop_pass.

    Embeddings that keep a float output are untouched, so they still route to the
    cadence quantized_embedding_byte lowering.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False
        for node in graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.embedding.default
        ):
            if _try_prequantize_embedding(exported_program, node):
                modified = True

        if modified:
            exported_program = constant_prop_pass(exported_program)

        return ExportedProgramPassResult(exported_program, modified)
