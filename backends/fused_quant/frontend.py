# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.fused_quant.pre_quantize_passes.fold_batch_norm import (
    FoldBatchNorm,
)
from executorch.backends.fused_quant.pre_quantize_passes.fuse_add_softmax_into_masked_softmax import (
    FuseAddSoftmaxIntoMaskedSoftmax,
)
from executorch.backends.fused_quant.pre_quantize_passes.replace_batch_norm_with_conv import (
    ReplaceBatchNormWithConv,
)
from executorch.backends.fused_quant.quantizer.quantizer import FusedQuantQuantizer
from torch._export.utils import _detect_fake_mode_from_gm
from torch._inductor.decomposition import remove_decompositions
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


@torch.no_grad()
def trace(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    ops_to_keep: list[torch._ops.OpOverload] | None = None,
) -> ExportedProgram:
    """Export ``model`` to core ATen, holding ``ops_to_keep`` back from decomposition.

    The quantizer matches whole ATen ops, so any op it owns must survive
    ``run_decompositions`` intact -- pass ``quantizer.preserved_ops()`` here.
    ``aten._safe_softmax`` is always kept: it is what SDPA decomposes to, and
    :class:`FuseAddSoftmaxIntoMaskedSoftmax` needs it whole to recompose.
    """
    model.eval()
    decomp_table = torch.export.default_decompositions()
    remove_decompositions(
        decomp_table,
        [*(ops_to_keep or []), torch.ops.aten._safe_softmax.default],
    )
    return torch.export.export(model, inputs, strict=True).run_decompositions(
        decomp_table
    )


def prepare(program: ExportedProgram, quantizer: FusedQuantQuantizer) -> GraphModule:
    """Run the pre-quantization rewrites, then insert observers.

    The rewrites run here rather than in the caller because they must happen
    after tracing (they match on ATen ops) but before annotation (the quantizer
    must see the rewritten ops). Which ones run is derived from the quantizer:
    a pass is only useful if the quantizer can consume what it produces.
    """
    ops_to_keep = quantizer.preserved_ops()

    program = FoldBatchNorm()(program).exported_program

    # Any batch norm not folded into an upstream conv/linear becomes a
    # convolution instead, so it can be quantized -- but only if the quantizer
    # actually handles convolution.
    if torch.ops.aten.convolution.default in ops_to_keep:
        program = ReplaceBatchNormWithConv()(program).exported_program

    # Recompose add(mask) + softmax into a single _masked_softmax before
    # annotation, so the very-negative mask sentinel never enters the quantized
    # domain. Only worth doing if the quantizer matches _masked_softmax.
    if torch.ops.aten._masked_softmax.default in ops_to_keep:
        program = FuseAddSoftmaxIntoMaskedSoftmax()(program).exported_program

    return prepare_pt2e(program.module(), quantizer)


def convert(
    gm: GraphModule, *, fold_quantize_into_mutable_buffers: bool = False
) -> GraphModule:
    """Remove observers and insert quantize/dequantize nodes."""
    gm = convert_pt2e(
        gm,
        fold_quantize_into_mutable_buffers=fold_quantize_into_mutable_buffers,
    )
    _normalize_affine_overloads(gm)
    # convert_pt2e rewrites the graph without refreshing meta["val"], so shape and
    # dtype metadata can be stale on the nodes it touched. Downstream passes read
    # it, so re-propagate before handing the graph on.
    fake_inputs = [node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]
    FakeTensorProp(
        gm, mode=_detect_fake_mode_from_gm(gm)
    ).propagate_dont_convert_inputs(*fake_inputs)
    return gm


def _normalize_affine_overloads(gm: GraphModule) -> None:
    """Canonicalize torchao affine quant nodes to their ``.default`` overload.

    ``AffineQuantizedObserverBase.convert`` (torchao) inserts ``quantize_affine`` /
    ``dequantize_affine`` using the ``OpOverloadPacket`` rather than the
    ``.default`` ``OpOverload``, leaving ``call_function`` targets that the fusion
    path (which matches ``.default``) and ``to_edge`` do not recognize.
    """
    replacements = {
        torch.ops.torchao.quantize_affine: torch.ops.torchao.quantize_affine.default,
        torch.ops.torchao.dequantize_affine: torch.ops.torchao.dequantize_affine.default,
    }
    changed = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in replacements:
            node.target = replacements[node.target]
            changed = True
    if changed:
        gm.recompile()
