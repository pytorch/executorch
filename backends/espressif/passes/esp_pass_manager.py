# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from executorch.exir._program_utils import _get_updated_graph_signature
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.pass_manager import ExportedProgramPassManager
from torch.fx import GraphModule

from .aten_to_esp_pass import AtenToEspPass


class _EspLoweringPass(ExportedProgramPassBase):
    def call(self, exported_program):
        if any(
            node.target
            in (exir_ops.edge.aten.addmm.default, exir_ops.edge.aten.mm.default)
            for node in exported_program.graph.nodes
        ):
            graph = copy.deepcopy(exported_program.graph)
            for original, cloned in zip(exported_program.graph.nodes, graph.nodes):
                cloned.name = original.name
            module = GraphModule(exported_program.graph_module, graph)
            module.meta = exported_program.graph_module.meta.copy()
            exported_program._graph_module = module
            exported_program._graph_signature = copy.deepcopy(
                exported_program.graph_signature
            )
            exported_program._state_dict = exported_program.state_dict.copy()
            exported_program._constants = exported_program.constants.copy()
        result = AtenToEspPass(exported_program)(exported_program.graph_module)
        exported_program._graph_signature = _get_updated_graph_signature(
            exported_program.graph_signature, result.graph_module
        )
        return ExportedProgramPassResult(exported_program, result.modified)


class EspPassManager(ExportedProgramPassManager):
    """Run Espressif operator-library transformations on an Edge program."""

    def __init__(self) -> None:
        super().__init__(passes=[_EspLoweringPass(), RemoveUnusedConstantsPass()])
