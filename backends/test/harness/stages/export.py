from typing import Any, Optional, Tuple

import torch

from executorch.backends.test.harness.stages.stage import Stage, StageType
from torch.export import export, ExportedProgram


class Export(Stage):
    def __init__(self, dynamic_shapes: Optional[Tuple[Any]] = None):
        self.exported_program = None
        self.dynamic_shapes = dynamic_shapes

    def stage_type(self) -> StageType:
        return StageType.EXPORT

    def run(
        self,
        artifact: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
    ) -> None:
        self.exported_program = export(
            artifact, inputs, dynamic_shapes=self.dynamic_shapes, strict=True
        )

    @property
    def artifact(self) -> ExportedProgram:
        return self.exported_program

    @property
    def graph_module(self) -> str:
        return self.exported_program.graph_module
