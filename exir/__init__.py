# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from executorch.exir.capture import (
    _capture_legacy_do_not_use,
    CallSpec,
    capture,
    capture_multiple,
    CaptureConfig,
    EdgeCompileConfig,
    ExecutorchBackendConfig,
)
from executorch.exir.emit import emit_program, EmitterOutput
from executorch.exir.program import (
    _to_edge,
    edge_to_executorch_passes,
    EdgeProgramManager,
    ExecutorchProgram,
    ExecutorchProgramManager,
    ExirExportedProgram,
    multi_method_program_to_executorch,
    MultiMethodExecutorchProgram,
    MultiMethodExirExportedProgram,
    to_edge,
)
from executorch.exir.tracer import ExirDynamoConfig
from torch.export import ExportedProgram, ExportGraphSignature

Value = Any

__all__ = [
    "emit_program",
    "EmitterOutput",
    "capture",
    "capture_multiple",
    "_capture_legacy_do_not_use",
    "CallSpec",
    "ExportedProgram",
    "ExirExportedProgram",
    "ExecutorchProgram",
    "ExportGraphSignature",
    "_to_edge",
    "to_edge",
    "EdgeProgramManager",
    "ExecutorchProgramManager",
    "edge_to_executorch_passes",
    "MultiMethodExirExportedProgram",
    "MultiMethodExecutorchProgram",
    "CaptureConfig",
    "EdgeCompileConfig",
    "ExecutorchBackendConfig",
    "Value",
    "multi_method_program_to_executorch",
    "ExirDynamoConfig",
]
