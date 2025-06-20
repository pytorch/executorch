import sys

from typing import Optional

from executorch.backends.test.harness.stages.stage import Stage, StageType
from executorch.exir import (
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
)
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.print_program import pretty_print, print_program


class ToExecutorch(Stage):
    def __init__(
        self,
        config: Optional[ExecutorchBackendConfig] = None,
    ):
        self.config = config or ExecutorchBackendConfig(
            extract_delegate_segments=True,
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
        self.executorch_program = None

    def stage_type(self) -> StageType:
        return StageType.TO_EXECUTORCH

    def run(self, artifact: EdgeProgramManager, inputs=None):
        self.executorch_program = artifact.to_executorch(self.config)

    @property
    def artifact(self) -> ExecutorchProgramManager:
        return self.executorch_program

    @property
    def graph_module(self) -> str:
        return self.executorch_program().graph_module

    def dump_artifact(self, path_to_dump: Optional[str]):
        """
        dump_artifact is overridden to dump the serialized program
        """
        original_stdout = sys.stdout

        sys.stdout = open(path_to_dump, "a") if path_to_dump else sys.stdout
        print(self.stage_banner() + "\n")
        pretty_print(self.artifact._emitter_output.program)
        print_program(
            self.artifact._emitter_output.program,
            show_meminfo=True,
            mark_dynamic_shape_tensor=True,
        )
        sys.stdout = original_stdout
