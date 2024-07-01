# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, List

from executorch.exir import ExirExportedProgram
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


@final
class ExecutorShardedBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        return PreprocessResult(
            processed_bytes=ExirExportedProgram(
                exported_program=edge_program,
                # Indicates that edge_program is already in edge dialect.
                after_to_edge_passes=True,
            )
            .to_executorch()
            .buffer,
        )
