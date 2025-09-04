# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from executorch.backends.nxp.edge_passes.move_auxiliary_operator_into_separate_qdq_cluster_pass import (
    MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass,
    MoveTrailingAuxiliaryOperatorIntoSeparateQDQClusterPass,
)
from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass
from executorch.exir import EdgeProgramManager
from executorch.exir.program._program import (
    _get_updated_graph_signature,
    _get_updated_range_constraints,
)

from torch import nn
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager


class NeutronEdgePassManager(PassManager):

    def __init__(self, passes: list[NeutronEdgePass] = None):
        passes: list[NeutronEdgePass] = passes or [
            MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass(),
            MoveTrailingAuxiliaryOperatorIntoSeparateQDQClusterPass(),
        ]

        super().__init__(
            passes,
            steps=10,  # Empirical value. At most 10 cycles of passes will be run.
        )

    def _transform_graph_module(self, module: nn.Module) -> PassResult:
        """Apply the passes to a single graph module."""
        pass_result: PassResult = super().__call__(module)

        graph_module = pass_result.graph_module
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return pass_result

    def __call__(self, epm: EdgeProgramManager) -> EdgeProgramManager:
        """Apply the passes to all graph modules in the edge program."""
        new_programs: dict[str, ExportedProgram] = {}

        for name, program in epm._edge_programs.items():
            pass_result = self._transform_graph_module(program.graph_module)

            if pass_result.modified:
                # Create a new exported program.
                new_program = ExportedProgram(
                    root=pass_result.graph_module,
                    graph=pass_result.graph_module.graph,
                    graph_signature=_get_updated_graph_signature(
                        program.graph_signature, pass_result.graph_module
                    ),
                    state_dict=program.state_dict,
                    range_constraints=_get_updated_range_constraints(
                        pass_result.graph_module
                    ),
                    module_call_graph=copy.deepcopy(program._module_call_graph),
                    example_inputs=program.example_inputs,
                    constants=program.constants,
                    verifiers=[program.verifier],
                )
                new_program.graph_module.meta.update(program.graph_module.meta)
                new_program.graph_module.meta.update(pass_result.graph_module.meta)

            else:
                # Keep the old exported program.
                new_program = program

            new_programs[name] = new_program

        if len(new_programs) == 0:
            # No passes were run, return the old EdgeProgramManager.
            return epm

        else:
            # Return a new EdgeProgramManager with the updated programs.
            return EdgeProgramManager(
                new_programs, copy.deepcopy(epm._config_methods), epm.compile_config
            )
