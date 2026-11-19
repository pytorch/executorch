# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.pass_manager import ExportedProgramPassManager, PassType
from torch.export import ExportedProgram

logger: logging.Logger = logging.getLogger(__name__)


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
        self._pm = ExportedProgramPassManager(passes, steps=1)

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        modified = False
        for _ in range(self._steps):
            result = self._pm(exported_program)
            exported_program = result.exported_program
            if not result.modified:
                return ExportedProgramPassResult(exported_program, modified)
            modified = True
        # A single-step group runs its passes once by design; only a group meant
        # to iterate to a fixed point can fail to converge.
        if self._steps > 1:
            logger.warning(
                "%s did not converge within %d steps; the graph was still changing "
                "on the last step",
                type(self).__name__,
                self._steps,
            )
        return ExportedProgramPassResult(exported_program, modified)
