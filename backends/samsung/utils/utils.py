# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from executorch.backends.transforms.utils import is_param_node
from executorch.exir.backend.backend_details import CompileSpec

from torch.export.exported_program import ExportedProgram


def get_compile_spec(
    compile_specs: List[CompileSpec], spec_name: str, required=False
) -> CompileSpec:
    for spec in compile_specs:
        if spec_name == spec.key:
            return spec
    assert not required, f"Require {spec_name} but it doesn't exist."


def is_graph_input(exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
    return node.op == "placeholder" and not is_param_node(exported_program, node)


def is_graph_output(node: torch.fx.Node) -> bool:
    # skip getitem node
    for user in node.users.keys():
        if user.op == "output" or (
            user.target.__name__ == "getitem" and is_graph_output(user)
        ):
            return True
    return False
