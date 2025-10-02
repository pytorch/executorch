from typing import Callable, List, Optional, Type, Union

from executorch.backends.test.harness.stages.stage import Stage, StageType
from executorch.exir import EdgeProgramManager
from executorch.exir.program._program import _transform
from torch._export.pass_base import PassType
from torch.export import ExportedProgram


class RunPasses(Stage):
    def __init__(
        self,
        pass_manager_cls: Type,
        pass_list: Optional[List[Type[PassType]]] = None,
        pass_functions: Optional[List[Callable]] = None,
    ):
        self.pass_manager_cls = pass_manager_cls
        self.pass_list = pass_list
        self.pass_functions = pass_functions
        self.edge_or_aten_program = None

    def stage_type(self) -> StageType:
        return StageType.RUN_PASSES

    def run(
        self, artifact: Union[EdgeProgramManager, ExportedProgram], inputs=None
    ) -> None:
        if isinstance(artifact, EdgeProgramManager):
            self.edge_or_aten_program = artifact
            if self.pass_list:
                pass_manager = self.pass_manager_cls(
                    artifact.exported_program(), self.pass_list
                )
                self.edge_or_aten_program._edge_programs["forward"] = (
                    pass_manager.transform()
                )
            if self.pass_functions:
                assert isinstance(self.pass_functions, list)
                for pass_function in self.pass_functions:
                    self.edge_or_aten_program._edge_programs["forward"] = pass_function(
                        self.edge_or_aten_program.exported_program()
                    )
        else:
            transformed_ep = artifact
            if self.pass_list:
                assert isinstance(self.pass_list, list)
                for pass_ in self.pass_list:
                    transformed_ep = _transform(transformed_ep, pass_())

            if self.pass_functions:
                assert isinstance(self.pass_functions, list)
                for pass_function in self.pass_functions:
                    transformed_ep = pass_function(transformed_ep)

            self.edge_or_aten_program = transformed_ep

    @property
    def artifact(self) -> Union[EdgeProgramManager, ExportedProgram]:
        return self.edge_or_aten_program

    @property
    def graph_module(self) -> str:
        if isinstance(self.edge_or_aten_program, EdgeProgramManager):
            return self.edge_or_aten_program.exported_program().graph_module
        else:
            return self.edge_or_aten_program.graph_module
