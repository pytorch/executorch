# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, cast, List, Optional, Type

import torch
import torch.fx
import torch.utils._pytree as pytree
from executorch.backends.cadence.aot.fuse_ops import (
    CadenceFuseOpsInGraph,
    FuseFullThenReshapePass,
    FuseSliceSameDimPass,
    FuseTransposeOrPermuteOpPairsPass,
)
from executorch.backends.cadence.aot.pass_utils import CompileMode, EdgePassesConfig
from executorch.backends.cadence.aot.remove_ops import (
    CadenceRemoveNops,
    RemoveAliasCopyOpPass,
    RemoveCloneOpsTransformImported,
    RemoveDetachCopyPass,
    RemoveNopExpandOpPass,
    RemoveNopSliceOrViewOpPass,
    RemovePermutesAroundElementwiseOps,
    RemoveRedundantOps,
    RemoveToOpsPass,
    RemoveZeroSizedCatArgsPass,
)
from executorch.backends.cadence.aot.reorder_ops import CadenceReorderOpsInGraph
from executorch.backends.cadence.aot.replace_ops import (
    CadenceReplaceOpsInGraph,
    ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass,
    ReplaceAtenAvgPoolWithCadenceAvgPoolPass,
    ReplaceAtenConvolutionWithCadenceConvolutionPass,
    ReplaceAtenLinalgSvdWithCadenceLinalgSvdPass,
    ReplaceConvolutionOptionalArgsWithConcreteArgsPass,
    ReplaceConvWithIm2RowAndLinear,
    ReplaceFullLikeWithFullPass,
    ReplaceFunctionallyEquivalentOpTargets,
    ReplaceInfArgInFullWithValuePass,
    ReplaceLogicalNotBooleanWhereWithWherePass,
    ReplaceMatmulWithTransposedMatmulPass,
    ReplaceMMWithAddMMPass,
    ReplaceMulTensorWithMulAndFullOpsPass,
    ReplacePT2DequantWithCadenceDequantPass,
    ReplacePT2QuantWithCadenceQuantPass,
    ReplaceRepeatWithCatPass,
    ReplaceSafeSoftmaxWithSoftmax,
    ReplaceScalarTensorWithFullPass,
    ReplaceScalarWithTensorArgPass,
    ReplaceSqueezeAndUnsqueezeWithViewPass,
    ReplaceTorchQuantizedEmbeddingWithCadenceQuantizedEmbedding,
)
from executorch.backends.cadence.aot.simplify_ops import (
    BindOptionalArgsPass,
    CadenceSimplifyOpsInGraph,
    SimplifySliceOpPass,
)
from executorch.backends.cadence.aot.type_dispatch import CompileTimeTypeDispatchPass
from executorch.exir import EdgeProgramManager
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.scalar_to_tensor_pass import ScalarToTensorPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.pass_base import PassBase


class InitializePipeline(ExportPass):
    """
    Initialize the pass pipeline. This should invariably be the first pass to
    run.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        dead_code_elimination_pass(graph_module)
        result = SpecPropPass()(graph_module)
        assert result is not None
        return result


class FinalizePipeline(ExportPass):
    """
    The final cleanup pass after running the pass pipeline.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        finalize_passes: List[PassType] = [
            ScalarToTensorPass(),
            SpecPropPass(),
        ]
        result = PassManager(passes=finalize_passes)(graph_module)
        dead_code_elimination_pass(result.graph_module)
        return result


# Similar to what's done in executorch/exir/pass_base.py
Argument = Any  # pyre-ignore


# The passes that must run for the graph to be legal on the target, regardless
# of compile mode. Everything else in the pipeline is an optimization.
REQUIRED_PASSES: frozenset[Type[PassBase]] = frozenset(
    {
        InitializePipeline,
        FinalizePipeline,
        FuseSliceSameDimPass,
        RemoveAliasCopyOpPass,
        RemoveCloneOpsTransformImported,
        RemoveDetachCopyPass,
        RemoveNopExpandOpPass,
        RemoveToOpsPass,
        RemoveZeroSizedCatArgsPass,
        ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass,
        ReplaceAtenAvgPoolWithCadenceAvgPoolPass,
        ReplaceAtenConvolutionWithCadenceConvolutionPass,
        ReplaceAtenLinalgSvdWithCadenceLinalgSvdPass,
        ReplaceConvolutionOptionalArgsWithConcreteArgsPass,
        ReplaceFullLikeWithFullPass,
        ReplaceFunctionallyEquivalentOpTargets,
        ReplaceInfArgInFullWithValuePass,
        ReplaceLogicalNotBooleanWhereWithWherePass,
        ReplaceMatmulWithTransposedMatmulPass,
        ReplaceMMWithAddMMPass,
        ReplacePT2DequantWithCadenceDequantPass,
        ReplacePT2QuantWithCadenceQuantPass,
        ReplaceRepeatWithCatPass,
        ReplaceScalarTensorWithFullPass,
        ReplaceScalarWithTensorArgPass,
        ReplaceSqueezeAndUnsqueezeWithViewPass,
        ReplaceTorchQuantizedEmbeddingWithCadenceQuantizedEmbedding,
        BindOptionalArgsPass,
        SimplifySliceOpPass,
    }
)


def _get_pipeline() -> list[Type[PassBase]]:
    """The full ordered pass pipeline.

    Order is load-bearing and levels are interleaved, so this list is the single
    source of truth: modes are expressed by removing entries from it, never by
    reordering or concatenating.
    """
    passes = [
        InitializePipeline,
        RemoveRedundantOps.passes,
        CadenceReorderOpsInGraph.passes,
        # Phase ordering: remove -> fusion -> replacement passes.
        CadenceRemoveNops.passes,
        CadenceFuseOpsInGraph.passes,
        CadenceReplaceOpsInGraph.passes,
        CadenceSimplifyOpsInGraph.passes,
        FinalizePipeline,
        FuseFullThenReshapePass,
        RemovePermutesAroundElementwiseOps,
        FuseTransposeOrPermuteOpPairsPass,
        RemoveNopSliceOrViewOpPass,
        CompileTimeTypeDispatchPass,
    ]
    return pytree.tree_flatten(passes)[0]


def get_passes(
    mode: CompileMode | str,
    edge_passes_config: Optional[EdgePassesConfig] = None,
) -> list[Type[PassBase]]:
    # Coerce at the choke point: modes arrive from CLIs and JSON as plain
    # strings, and a str silently matches none of the checks below.
    mode = CompileMode(mode)
    config = edge_passes_config or EdgePassesConfig()
    passes = _get_pipeline()
    if mode is CompileMode.MINIMAL:
        passes = [p for p in passes if p in REQUIRED_PASSES]
    if mode is not CompileMode.SIZE:
        passes = [p for p in passes if p is not CompileTimeTypeDispatchPass]
    if not config.use_im2row_transform:
        passes = [p for p in passes if p is not ReplaceConvWithIm2RowAndLinear]
    return passes


def apply_exir_ops_passes(
    mode: CompileMode,
    edge_prog_manager: EdgeProgramManager,
    edge_passes_config: Optional[EdgePassesConfig] = None,
) -> EdgeProgramManager:
    cadence_passes = [
        (lambda graph_module, p=p: p()(graph_module))
        for p in get_passes(mode, edge_passes_config)
    ]
    cadence_prog_manager = edge_prog_manager.transform(
        cast(
            list[Callable[[torch.fx.GraphModule], Optional[PassResult]]], cadence_passes
        )
    )
    return cadence_prog_manager


def apply_torch_ops_passes(expo_program: ExportedProgram) -> ExportedProgram:
    """
    Applies compiler passes on torch.ops IR, including torch.ops.aten, torch.ops.cadence, etc.
    expo_program is expected to be the output of the torch.export.export().
    """

    aten_passes: List[Callable[[torch.fx.GraphModule], Optional[PassResult]]] = [
        ReplaceSafeSoftmaxWithSoftmax(),
        ReplaceMulTensorWithMulAndFullOpsPass(),
    ]
    # TODO(T230417247): Use PassResult which is currently ignored.
    PassManager(aten_passes)(expo_program.graph_module)
    return expo_program
