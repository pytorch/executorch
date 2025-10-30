from typing import Optional

from executorch.backends.test.harness.stages.stage import Stage, StageType
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from torch.export import ExportedProgram


class ToEdge(Stage):
    def __init__(self, edge_compile_config: Optional[EdgeCompileConfig] = None):
        self.edge_compile_conf = edge_compile_config or EdgeCompileConfig()
        self.edge_dialect_program = None

    def stage_type(self) -> StageType:
        return StageType.TO_EDGE

    def run(self, artifact: ExportedProgram, inputs=None) -> None:
        self.edge_dialect_program = to_edge(
            artifact, compile_config=self.edge_compile_conf
        )

    @property
    def artifact(self) -> EdgeProgramManager:
        return self.edge_dialect_program

    @property
    def graph_module(self) -> str:
        return self.edge_dialect_program.exported_program().graph_module
