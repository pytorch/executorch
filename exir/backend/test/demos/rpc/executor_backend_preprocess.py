# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, List

import executorch.exir as exir
from executorch.exir.backend.backend_details import BackendDetails, ExportedProgram
from executorch.exir.backend.compile_spec_schema import CompileSpec


@final
class ExecutorBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> bytes:
        new_prog = edge_program.transform(
            *exir.edge_to_executorch_passes(exir.ExecutorchBackendConfig())
        )
        program = exir.emit_program(new_prog).program
        buffer = exir.serialize_to_flatbuffer(program)
        return buffer
