# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT backend implementation for ExecuTorch."""

from typing import final, List

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram


@final
class TensorRTBackend(BackendDetails):
    """TensorRT backend for accelerating models on NVIDIA GPUs.

    This backend compiles ExecuTorch edge programs to TensorRT engines
    for optimized inference on NVIDIA hardware.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Compile edge program to TensorRT engine.

        Args:
            edge_program: The edge dialect program to compile.
            compile_specs: Backend-specific compilation options.

        Returns:
            PreprocessResult containing the serialized TensorRT engine.
        """

        return PreprocessResult(processed_bytes=b"")
