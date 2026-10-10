# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

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
from torch._inductor.decomposition import remove_decompositions
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e


def trace(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    quantizer: FusedQuantQuantizer,
    *,
    is_qat: bool = False,
) -> ExportedProgram:
    """Export ``model`` to core ATen and apply the pre-quantization rewrites.

    The quantizer matches whole ATen ops, so every op in
    ``quantizer.preserved_ops()`` is held back from ``run_decompositions``.
    ``aten._safe_softmax`` is always kept: it is what SDPA decomposes to, and
    :class:`FuseAddSoftmaxIntoMaskedSoftmax` needs it whole to recompose.

    The rewrites must run after decomposition (they match on ATen ops) but
    before annotation (the quantizer must see the rewritten ops). Which ones run
    is derived from the quantizer: a pass is only useful if the quantizer can
    consume what it produces.

    With ``is_qat`` the model is exported in training mode and batch norms are
    kept, so the result can go to ``prepare_qat_pt2e``; otherwise it is exported
    in eval mode with batch norms folded. Either way the caller's model is left
    in the mode it was passed in.
    """
    ops_to_keep = quantizer.preserved_ops()

    decomp_table = torch.export.default_decompositions()
    remove_decompositions(
        decomp_table,
        [*ops_to_keep, torch.ops.aten._safe_softmax.default],
    )
    was_training = model.training
    model.train(is_qat)
    try:
        with contextlib.nullcontext() if is_qat else torch.no_grad():
            program = torch.export.export(
                model, inputs, strict=True
            ).run_decompositions(decomp_table)
    finally:
        model.train(was_training)

    if is_qat:
        # Batch norms train with the model; folding them is a post-training step.
        if torch.ops.aten._masked_softmax.default in ops_to_keep:
            program = FuseAddSoftmaxIntoMaskedSoftmax()(program).exported_program
        return program

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

    return program


def prepare(program: ExportedProgram, quantizer: FusedQuantQuantizer) -> GraphModule:
    """Annotate ``program`` with ``quantizer`` and insert observers."""
    return prepare_pt2e(program.module(), quantizer)
