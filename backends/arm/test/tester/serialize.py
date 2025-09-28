# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional

import executorch.backends.xnnpack.test.tester.tester as tester

import torch.fx

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec

from executorch.backends.arm.test.runner_utils import (
    get_elf_path,
    get_target_board,
    run_target,
)

from executorch.exir import ExecutorchProgramManager
from torch.utils._pytree import tree_flatten


logger = logging.getLogger(__name__)


class Serialize(tester.Serialize):
    def __init__(
        self,
        compile_spec: ArmCompileSpec,
        module: Optional[torch.nn.Module],
        timeout: int = 120,
    ):
        """
        Args:
            compile_spec: CompileSpecs to be used for serialization.
            module: Original Module to be used for serialization. Optional - can be used for reference output generation.
            timeout: Timeout for fvp. Default is 120 seconds.
        """
        super().__init__()
        self.module = module
        self.timeout = timeout
        self.executorch_program_manager: ExecutorchProgramManager | None
        self.compile_spec = compile_spec

    def run(self, artifact: ExecutorchProgramManager, inputs=None) -> None:
        super().run(artifact, inputs)
        # Keep the entire ExecutorchProgramManager for execution.
        self.executorch_program_manager = artifact

    def run_artifact(self, inputs):
        if self.executorch_program_manager is None:
            raise RuntimeError(
                "Tried running artifact from Serialize stage without running the stage."
            )
        inputs_flattened, _ = tree_flatten(inputs)
        intermediate_path = self.compile_spec.get_intermediate_path()
        target_board = get_target_board(self.compile_spec)
        elf_path = get_elf_path(target_board)

        if not os.path.exists(elf_path):
            raise FileNotFoundError(
                f"Did not find build arm_executor_runner in path {elf_path}, run setup_testing.sh?"
            )

        return run_target(
            self.executorch_program_manager,
            inputs_flattened,
            intermediate_path,
            target_board,
            elf_path,
            self.timeout,
        )
