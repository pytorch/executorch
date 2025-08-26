# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import executorch.exir as exir
import torch
from executorch.exir import EdgeCompileConfig, ExportedProgram
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.program._program import (
    to_edge_transform_and_lower,
)


def to_edge_transform_and_lower_to_enn(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    custom_pass_config: List[PassType] = None,
    compile_specs: Optional[CompileSpec] = None,
) -> exir.ExecutorchProgramManager:
    assert compile_specs is not None, "For now, we must deliver complile specs"
    prog = torch.export.export(module, inputs)
    return to_edge_transform_and_lower(
        prog,
        compile_config=[],
    )
