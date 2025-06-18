from executorch.backends.test.harness.stages.stage import Stage, StageType
from executorch.exir import EdgeProgramManager
from executorch.exir.backend.backend_api import validation_disabled
from executorch.exir.backend.partitioner import Partitioner


class Partition(Stage):
    def __init__(self, partitioner: Partitioner):
        self.partitioner = partitioner
        self.delegate_module = None

    def stage_type(self) -> StageType:
        return StageType.PARTITION

    def run(self, artifact: EdgeProgramManager, inputs=None):
        with validation_disabled():
            self.delegate_module = artifact
            self.delegate_module = self.delegate_module.to_backend(self.partitioner)

    @property
    def artifact(self) -> EdgeProgramManager:
        return self.delegate_module

    @property
    def graph_module(self) -> str:
        return self.delegate_module.exported_program().graph_module
