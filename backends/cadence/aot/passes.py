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
    FuseTransposeOrPermuteOpPairsPass,
)
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    create_cadence_pass_filter,
    register_cadence_pass,
)

from executorch.backends.cadence.aot.remove_ops import (
    CadenceRemoveNops,
    RemoveNopSliceOrViewOpPass,
    RemoveRedundantOps,
)
from executorch.backends.cadence.aot.reorder_ops import CadenceReorderOpsInGraph
from executorch.backends.cadence.aot.replace_ops import (
    CadenceReplaceOpsInGraph,
    ReplaceMulTensorWithMulAndFullOpsPass,
)
from executorch.backends.cadence.aot.simplify_ops import CadenceSimplifyOpsInGraph
from executorch.backends.cadence.aot.type_dispatch import CompileTimeTypeDispatchPass
from executorch.exir import EdgeProgramManager
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.scalar_to_tensor_pass import ScalarToTensorPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from torch.export.exported_program import ExportedProgram


@register_cadence_pass(CadencePassAttribute(opt_level=0))
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


@register_cadence_pass(CadencePassAttribute(opt_level=0))
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


def get_passes_in_default_order() -> list[Type[ExportPass]]:
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
        FuseTransposeOrPermuteOpPairsPass,
        RemoveNopSliceOrViewOpPass,
        CompileTimeTypeDispatchPass,
    ]
    return pytree.tree_flatten(passes)[0]


def apply_exir_ops_passes(
    opt_level: int,
    edge_prog_manager: EdgeProgramManager,
) -> EdgeProgramManager:
    passes = get_passes_in_default_order()
    pass_filter = create_cadence_pass_filter(opt_level)
    cadence_passes = [
        (
            lambda graph_module, filtered_pass=filtered_pass: filtered_pass()(
                graph_module
            )
        )
        for filtered_pass in list(filter(pass_filter, passes))
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
        ReplaceMulTensorWithMulAndFullOpsPass()
    ]
    # TODO(T230417247): Use PassResult which is currently ignored.
    PassManager(aten_passes)(expo_program.graph_module)
    return expo_program
