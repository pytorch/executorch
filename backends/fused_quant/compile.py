# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Optional

import torch
from executorch.backends.fused_quant.frontend import convert, prepare, trace
from executorch.backends.fused_quant.passes import get_fused_quant_passes
from executorch.backends.fused_quant.quantizer.defaults import (
    make_fused_quant_quantizer,
)
from executorch.backends.fused_quant.quantizer.quantizer import FusedQuantQuantizer
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from executorch.exir.pass_manager import ExportedProgramPassManager, PassType


def compile_to_fused_quant(
    model: torch.nn.Module,
    inputs: Sequence[tuple[torch.Tensor, ...]],
    quantizer: Optional[FusedQuantQuantizer] = None,
    *,
    passes: Optional[Sequence[PassType]] = None,
    fold_quantize_into_mutable_buffers: bool = False,
) -> EdgeProgramManager:
    """Quantize, fuse, and optimize ``model`` into the fused_quant edge graph.

    Runs trace -> prepare -> calibrate -> convert -> fuse -> export -> to_edge ->
    ``passes``, and stops there. The result is backend-independent: a mixture of
    ATen and fused_quant ops in edge dialect, before any coloring or lowering.
    This is the seam a backend plugs into.

    From here a backend runs its own colorer, lowering, and partitioner. What it
    does not claim must be handled by ``DecomposeFusedQuant``, which turns the
    remaining fused_quant ops back into quantize/dequantize plus ATen; see
    ``test_compile.py`` for that tail written out.

    Args:
        inputs: The calibration set. ``inputs[0]`` is also the example input for
            tracing.
        quantizer: Defaults to :func:`make_fused_quant_quantizer`.
        passes: Edge passes to run after ``to_edge``. Defaults to
            :func:`get_fused_quant_passes`; pass an explicit list to use a
            different pipeline, or an empty one to skip optimization entirely.
    """
    if quantizer is None:
        quantizer = make_fused_quant_quantizer()
    if passes is None:
        passes = get_fused_quant_passes()

    program = trace(model, inputs[0], ops_to_keep=quantizer.preserved_ops())
    graph_module = prepare(program, quantizer)

    for inp in inputs:
        graph_module(*inp)

    graph_module = convert(
        graph_module,
        fold_quantize_into_mutable_buffers=fold_quantize_into_mutable_buffers,
    )

    # Fusion runs before export/to_edge, on ATen IR. The quantizer already
    # identified which ops to quantize, so fusion can match directly on the
    # quantized_decomposed ops that convert introduced.
    quantizer.fuse(graph_module)

    # _skip_dim_order keeps classic aten copies instead of dim_order_ops
    # (_to_dim_order_copy), which not every runtime registers.
    #
    # The preserved ops are held back from decomposition precisely because the
    # quantizer fuses them, and DecomposeFusedQuant re-emits them if a backend
    # does not claim the fused op. Several (aten.linear, aten.silu, ...) are not
    # core ATen, so the edge verifier needs to know they are expected.
    edge_program_manager = to_edge(
        torch.export.export(graph_module, inputs[0]),
        compile_config=EdgeCompileConfig(
            _skip_dim_order=True,
            _core_aten_ops_exception_list=list(quantizer.preserved_ops()),
        ),
    )
    return edge_program_manager.transform(ExportedProgramPassManager(list(passes)))
