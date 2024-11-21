# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, List, Optional, Type

import torch
import torch.fx
import torch.utils._pytree as pytree
from executorch.backends.cadence.aot.fuse_ops import (
    CadenceFuseOpsInGraph,
    FuseFullThenReshapePass,
    FuseTransposeOpPairsPass,
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
from executorch.backends.cadence.aot.replace_ops import CadenceReplaceOpsInGraph
from executorch.backends.cadence.aot.simplify_ops import CadenceSimplifyOpsInGraph
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.scalar_to_tensor_pass import ScalarToTensorPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass


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


def get_passes_in_default_order() -> List[Type[PassType]]:
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
        FuseTransposeOpPairsPass,
        RemoveNopSliceOrViewOpPass,
    ]
    return pytree.tree_flatten(passes)[0]


def get_cadence_passes(
    opt_level: int,
) -> List[Optional[PassResult]]:
    passes = get_passes_in_default_order()
    pass_filter = create_cadence_pass_filter(opt_level)
    filtered_passes = [
        # pyre-fixme[20]: Call `torch.fx.passes.infra.pass_base.PassBase.__call__` expects argument `graph_module`.
        filtered_pass()
        # pyre-fixme[6]: In call `filter.__new__` ... got `List[Type[typing.Callable[[GraphModule], Optional[PassResult]]]]`.
        for filtered_pass in list(filter(pass_filter, passes))
    ]
    return filtered_passes
