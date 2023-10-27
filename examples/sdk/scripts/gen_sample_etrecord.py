# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generate fixture files
from pathlib import Path

import executorch.exir as exir
from executorch.exir import ExecutorchBackendConfig

from executorch.exir.tests.models import BasicSinMax
from executorch.sdk.etrecord import generate_etrecord
from torch.export import export


def get_module_path() -> Path:
    return Path(__file__).resolve().parents[0]


def gen_etrecord():
    f = BasicSinMax()
    aten_dialect = export(
        f,
        f.get_random_inputs(),
    )
    edge_program = exir.to_edge(
        aten_dialect, compile_config=exir.EdgeCompileConfig(_check_ir_validity=False)
    )
    et_program = edge_program.to_executorch(ExecutorchBackendConfig(passes=[]))
    generate_etrecord(
        str(get_module_path()) + "/etrecord.bin",
        edge_dialect_program=edge_program,
        executorch_program=et_program,
        export_modules={
            "aten_dialect_output": aten_dialect,
        },
    )


if __name__ == "__main__":
    gen_etrecord()
