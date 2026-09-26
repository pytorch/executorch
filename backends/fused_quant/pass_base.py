# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.pass_manager import ExportedProgramPassManager, PassType
from torch.export import ExportedProgram


class IterativePassGroup(ExportedProgramPassBase):
    """Runs a group of passes as a single ExportedProgramPassBase.

    Useful for composing pass pipelines — e.g., running a set of optimization
    passes multiple times until convergence while other passes run once.
    """

    def __init__(self, passes: list[PassType], steps: int) -> None:
        # Retain the children and step count so instrumentation can recursively
        # rebuild this group with wrapped leaf passes.
        self._passes = passes
        self._steps = steps
        self._pm = ExportedProgramPassManager(passes, steps=steps)

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        result = self._pm(exported_program)
        return ExportedProgramPassResult(result.exported_program, result.modified)
