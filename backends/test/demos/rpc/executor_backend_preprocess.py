from typing import Callable, Dict, final, List

import torch
from executorch.backends.backend_details import BackendDetails
from executorch.backends.compile_spec_schema import CompileSpec
from executorch.exir import edge_dialect_to_executorch, EdgeDialectGraphModule


@final
class ExecutorBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_ir_module: EdgeDialectGraphModule,
        compile_specs: List[CompileSpec],
    ) -> bytes:
        return edge_dialect_to_executorch(edge_ir_module).buffer
