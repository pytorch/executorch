from typing import Any

from executorch.exir.capture import (
    capture,
    capture_multiple,
    CaptureConfig,
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ServerCompileConfig,
)
from executorch.exir.emit import emit_program, EmitterOutput
from executorch.exir.program import (
    _to_edge,
    edge_to_executorch_passes,
    ExecutorchProgram,
    ExirExportedProgram,
    multi_method_program_to_executorch,
    MultiMethodExecutorchProgram,
    MultiMethodExirExportedProgram,
)
from executorch.exir.serialize import serialize_to_flatbuffer
from executorch.exir.tracer import ExirDynamoConfig
from torch._export import (  # lots of people are doing from exir import CallSpec, ExportGraphSignature, ExportedProgram which seems wrong
    CallSpec,
    ExportedProgram,
    ExportGraphSignature,
)

Value = Any

__all__ = [
    "emit_program",
    "EmitterOutput",
    "capture",
    "capture_multiple",
    "CallSpec",
    "ExportedProgram",
    "ExirExportedProgram",
    "ExecutorchProgram",
    "ExportGraphSignature",
    "_to_edge",
    "edge_to_executorch_passes",
    "MultiMethodExirExportedProgram",
    "MultiMethodExecutorchProgram",
    "CaptureConfig",
    "EdgeCompileConfig",
    "ServerCompileConfig",
    "ExecutorchBackendConfig",
    "Value",
    "serialize_to_flatbuffer",
    "multi_method_program_to_executorch",
    "ExirDynamoConfig",
]
