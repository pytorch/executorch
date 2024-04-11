# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
from typing import Any, Dict, Optional, Union

from executorch.exir.capture import (
    _capture_legacy_do_not_use,
    CallSpec,
    capture,
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
    to_edge,
)
from executorch.exir.tracer import ExirDynamoConfig
from torch._export import _export_load_util, _export_save_util
from torch._export.serde.serialize import SerializedArtifact
from torch.export import ExportedProgram, ExportGraphSignature

Value = Any

__all__ = [
    "emit_program",
    "EmitterOutput",
    "capture",
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
    "CaptureConfig",
    "EdgeCompileConfig",
    "ExecutorchBackendConfig",
    "Value",
    "ExirDynamoConfig",
]


def save(
    ep: ExportedProgram,
    f: Union[str, os.PathLike, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    if not isinstance(ep, ExportedProgram):
        raise TypeError(f"save() expects an ExportedProgram but got {type(ep)}")

    from executorch.exir.serde.serialize import serialize

    artifact: SerializedArtifact = serialize(ep, opset_version)
    _export_save_util(f, artifact, extra_files)


def load(
    f: Union[str, os.PathLike, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ExportedProgram:

    extra_files = extra_files or {}
    artifact: SerializedArtifact = _export_load_util(f, extra_files)

    from executorch.exir.serde.serialize import deserialize

    # Deserialize ExportedProgram
    ep = deserialize(artifact, expected_opset_version)

    return ep
