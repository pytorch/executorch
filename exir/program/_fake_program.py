# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import Dict, Union

import torch

from torch._guards import detect_fake_mode
from torch.export import ExportedProgram


def get_fake_program(real_exported_program: ExportedProgram) -> ExportedProgram:
    """Create a fake exported program. This uses fake tensors for the state dict
    to prevent mutation, and points to the real constants, to avoid large memory
    usage from copying when constants are large.

    Args:
        real_exported_program: the original exported program
    Returns:
        A new exported program, with fake tensors.
    """
    fake_mode = detect_fake_mode(
        tuple(
            node.meta["val"]
            for node in real_exported_program.graph.nodes
            if node.op == "placeholder"
        )
    )
    if fake_mode is None:
        raise AssertionError(
            "Could not detect fake mode for graph: ", real_exported_program.graph
        )

    new_state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]] = {}

    for key, tensor in real_exported_program.state_dict.items():
        fake = fake_mode.from_tensor(tensor, static_shapes=True)
        new_state_dict[key] = fake

    gm = copy.deepcopy(real_exported_program.graph_module)
    fake_exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=copy.deepcopy(real_exported_program.graph_signature),
        state_dict=new_state_dict,
        range_constraints=copy.deepcopy(real_exported_program.range_constraints),
        module_call_graph=copy.deepcopy(real_exported_program.module_call_graph),
        constants=real_exported_program.constants,
        verifiers=[real_exported_program.verifier],
    )
    return fake_exported_program


def update_to_real_program(
    fake_exported_program: ExportedProgram, real_exported_program: ExportedProgram
) -> None:
    """Update the fake exported program to point to the real state dict. Modifies the
    fake exported program in-place.
    """
    for k, v in real_exported_program.state_dict.items():
        fake_exported_program._state_dict[k] = v
