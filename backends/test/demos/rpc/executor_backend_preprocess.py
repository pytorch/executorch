from typing import Callable, Dict, final, List

import torch
from executorch.backends.backend_details import BackendDetails
from executorch.backends.compile_spec_schema import CompileSpec
from executorch.exir import export_graph_module_to_executorch, ExportGraphModule


@final
class ExecutorBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_ir_module: ExportGraphModule,
        compile_specs: List[CompileSpec],
    ) -> bytes:
        return export_graph_module_to_executorch(edge_ir_module).buffer
