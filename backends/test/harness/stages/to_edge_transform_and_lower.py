from typing import List, Optional, Type

from executorch.backends.test.harness.stages.stage import Stage, StageType
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.partitioner import Partitioner

from torch.export import ExportedProgram


class ToEdgeTransformAndLower(Stage):
    def __init__(
        self,
        default_partitioner_cls: Type | None = None,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        self.partitioners = (
            partitioners or [default_partitioner_cls()]
            if default_partitioner_cls is not None
            else []
        )
        self.edge_compile_conf = edge_compile_config or EdgeCompileConfig(
            _check_ir_validity=False
        )
        self.edge_dialect_program = None

    def stage_type(self) -> StageType:
        return StageType.TO_EDGE_TRANSFORM_AND_LOWER

    def run(
        self, artifact: ExportedProgram, inputs=None, generate_etrecord: bool = False
    ) -> None:
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact,
            compile_config=self.edge_compile_conf,
            partitioner=self.partitioners,
            generate_etrecord=generate_etrecord,
        )

    @property
    def artifact(self) -> EdgeProgramManager:
        return self.edge_dialect_program

    @property
    def graph_module(self) -> str:
        return self.edge_dialect_program.exported_program().graph_module
