# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch.export import ExportedProgram


class ConstantFold(ExportedProgramPassBase):
    """Runs ``constant_prop_pass`` as a pass with a correct ``modified`` flag.

    ``constant_prop_pass`` folds pure-constant subgraphs but returns no signal for
    whether it changed the graph; this wrapper derives ``modified`` from the node
    count so it composes correctly in an iterative pass pipeline.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        n_before = len(exported_program.graph_module.graph.nodes)
        folded = constant_prop_pass(exported_program)
        modified = len(folded.graph_module.graph.nodes) != n_before
        return ExportedProgramPassResult(folded, modified)
