# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, List

from executorch.exir import ExirExportedProgram
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.runtime_info_schema import RuntimeInfo


@final
class ExecutorBackend(BackendDetails):
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

    @staticmethod
    def is_compatible(
        processed_bytes: bytes,
        compile_spec: List[CompileSpec],
        runtime_info: List[RuntimeInfo],
    ) -> bool:
        runtime_dict = {runtime.key: runtime.value for runtime in runtime_info}
        # This is optional, we don't necessarily need to fully deserialize the model to check binary version
        binary = deserialize_pte_binary(processed_bytes)
        current_binary_version = binary.version

        binary_version_matches = (
            int(runtime_dict["supported_binary_version"].decode("utf-8"))
            == current_binary_version
        )

        # Since we don't have a way to get the runtime version from c++, hardcoded for now
        runtime_version_is_compatible = runtime_dict["runtime_version"] == b"ET_12"

        return binary_version_matches and runtime_version_is_compatible
