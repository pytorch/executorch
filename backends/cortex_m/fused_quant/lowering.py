# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lowers the fused_quant edge program to cortex_m ops.

The cortex_m ops are ordinary kernels, so nothing is partitioned, and nothing
needs claiming either: lowering ahead of the shared fallback passes leaves them
only the fused nodes no kernel runs. ``get_cortex_m_passes`` runs after the
fused_quant passes:

1. ``FoldActivations`` folds each clamping activation into the fused op that
   feeds it, as that op's output range.
2. ``LowerToCortexM`` rewrites each fused_quant node a CMSIS-NN kernel can run,
   judged from its qparams, shapes and arguments, into its cortex_m op.
3. ``ToChannelsFirst`` and ``DecomposeFusedQuant`` fall the fused nodes left
   back to quantize / dequantize plus ATen, for the portable kernels.
4. ``RetargetToCortexM`` moves the int8 / int16 quantize and dequantize nodes,
   and the int8 permutes, to cortex_m.
"""

from typing import cast, Optional

import executorch.backends.cortex_m.fused_quant.ops  # noqa: F401
import executorch.backends.cortex_m.ops.operators  # noqa: F401
import torch
from executorch.backends.cortex_m.fused_quant import convolution, elementwise, linear
from executorch.backends.cortex_m.fused_quant.fold_activations import FoldActivations
from executorch.backends.cortex_m.fused_quant.utils import Context, Rules
from executorch.backends.cortex_m.passes.initialize_scratch_buffers_pass import (
    InitializeScratchBuffersPass,
)
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.transforms.fused_quant.decompose_fused_quant import (
    AssertNoFusedQuantOps,
    DecomposeFusedQuant,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.pass_manager import PassType
from torch import fx
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_base import PassBase, PassResult

RULES: Rules = {**elementwise.RULES, **linear.RULES, **convolution.RULES}

_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default
_RETARGETS = {
    _QUANTIZE: exir_ops.edge.cortex_m.quantize_per_tensor.default,
    _DEQUANTIZE: exir_ops.edge.cortex_m.dequantize_per_tensor.default,
}


class LowerToCortexM(ExportedProgramPassBase):
    """Rewrites each fused_quant node a CMSIS-NN kernel can run into its
    cortex_m op.

    Nodes are rewritten in graph order, so a rule that reads its producer,
    such as a table composing with the one ahead of it, sees it rewritten.
    """

    def __init__(self, target_config: CortexMTargetConfig) -> None:
        super().__init__()
        self.target_config = target_config

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        ctx = Context(exported_program, self.target_config)
        graph = exported_program.graph_module.graph
        modified = False
        for node in [node for node in graph.nodes if node.target in RULES]:
            rule = RULES[cast(EdgeOpOverload, node.target)]
            if rule.can_lower(ctx, node):
                rule.lower(ctx, node)
                modified = True
        if modified:
            exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, modified)


def _retarget(node: fx.Node) -> bool:
    """Move an int8 / int16 quantize, dequantize or int8 permute to cortex_m."""
    if node.target in _RETARGETS and len(node.args) == 6:
        quant_min, quant_max, dtype = cast(tuple[int, int, torch.dtype], node.args[3:])
        if dtype not in (torch.int8, torch.int16):
            return False
        info = torch.iinfo(dtype)
        if quant_min < info.min or quant_max > info.max:
            return False
        if node.kwargs.get("out_dtype") not in (None, torch.float32):
            return False
        node.target = _RETARGETS[node.target]
        node.kwargs = {}
        return True
    if node.target == _PERMUTE:
        inp, dims = cast(tuple[fx.Node, list[int]], node.args)
        val = inp.meta["val"]
        if val.dtype != torch.int8 or not 1 <= val.dim() <= 4:
            return False
        node.target = exir_ops.edge.cortex_m.transpose.default
        node.args = (inp, [d % val.dim() for d in dims])
        return True
    return False


class RetargetToCortexM(PassBase):
    """Moves the int8 / int16 quantize and dequantize nodes, and the int8
    permutes, to their cortex_m kernels: those around the lowered ops and those
    the fallback left."""

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            modified |= node.op == "call_function" and _retarget(node)
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)


def get_cortex_m_passes(
    exported_program: ExportedProgram,
    target_config: Optional[CortexMTargetConfig] = None,
) -> list[PassType]:
    """The passes that lower a fused_quant edge program to cortex_m ops.

    Run them after the fused_quant passes, with ``channels_last`` on so the
    convolutions and pools are in the layout of CMSIS-NN's NHWC kernels.
    Defaults to an M55, as ``CortexMPassManager`` does.
    """
    target_config = target_config or CortexMTargetConfig(cpu=CortexM.M55)
    return [
        FoldActivations(RULES, target_config),
        LowerToCortexM(target_config),
        ToChannelsFirst(),
        DecomposeFusedQuant(),
        RetargetToCortexM(),
        AssertNoFusedQuantOps(),
        InitializeScratchBuffersPass(exported_program, target_config),
        RemoveUnusedConstantsPass(),
    ]
