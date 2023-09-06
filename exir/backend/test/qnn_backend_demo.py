# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The fake qnnback
from typing import final, List

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


@final
class QnnBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        processed_bytes = "imqnncompiled"
        all_nodes_debug_handle = [
            node.meta["debug_handle"] for node in edge_program.graph.nodes
        ]
        return PreprocessResult(
            processed_bytes=bytes(processed_bytes, encoding="utf8"),
            # Assuming all nodes are fused as one op
            debug_handle_map={1: tuple(all_nodes_debug_handle)},
        )
