# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import executorch.exir as exir
import torch
from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.program._program import to_edge_transform_and_lower


def to_edge_transform_and_lower_to_enn(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    compile_specs: Optional[CompileSpec] = None,
) -> exir.ExecutorchProgramManager:
    assert (
        compile_specs is not None
    ), "Please provide compile specifications for enn backend"
    prog = torch.export.export(module, inputs)

    ahead_pass_list = [RemoveCloneOpsTransform()]
    return to_edge_transform_and_lower(
        prog,
        ahead_pass_list,
        {"forward": [EnnPartitioner(compile_specs)]},
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    )
