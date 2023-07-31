# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The fake qnnback
from typing import final, List

from executorch.backends.backend_details import BackendDetails, ExportedProgram
from executorch.backends.compile_spec_schema import CompileSpec


@final
class QnnBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> bytes:
        print("entering the lowerable parts in QnnBackend.preprocess....")
        processed_bytes = "imqnncompiled"
        return bytes(processed_bytes, encoding="utf8")
