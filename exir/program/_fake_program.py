# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

    from torch.export.exported_program import (
        ExportGraphSignature,
        ConstantArgument,
        OutputSpec,
        CustomObjArgument,
        InputSpec
    )
    def _get_updated_graph_signature(
            old_signature: ExportGraphSignature,
            new_gm: torch.fx.GraphModule,
        ) -> ExportGraphSignature:
            """
            Update the graph signature's user_input/user_outputs.
            """
            new_input_specs = []
            for i, node in enumerate(new_gm.graph.nodes):
                if node.op != "placeholder":
                    break

                assert i < len(
                    old_signature.input_specs
                ), "Number of inputs changed after transformation"
                old_input_spec = old_signature.input_specs[i]
                arg = (
                    old_input_spec.arg
                    if isinstance(
                        old_input_spec.arg, (ConstantArgument, CustomObjArgument)
                    )
                    else type(old_input_spec.arg)(node.name)
                )
                new_input_specs.append(
                    InputSpec(
                        old_input_spec.kind,
                        arg,
                        old_input_spec.target,
                        old_input_spec.persistent,
                    )
                )

            output_node = list(new_gm.graph.nodes)[-1]
            assert output_node.op == "output"

            new_output_specs = []
            for i, node in enumerate(output_node.args[0]):
                assert i < len(
                    old_signature.output_specs
                ), "Number of outputs changed after transformation"
                old_output_spec = old_signature.output_specs[i]
                arg = (
                    old_output_spec.arg
                    if isinstance(
                        old_output_spec.arg, (ConstantArgument, CustomObjArgument)
                    )
                    else type(old_output_spec.arg)(node.name)
                )
                new_output_specs.append(
                    OutputSpec(old_output_spec.kind, arg, old_output_spec.target)
                )

            new_signature = ExportGraphSignature(
                input_specs=new_input_specs, output_specs=new_output_specs
            )
            return new_signature
    gm = copy.deepcopy(real_exported_program.graph_module)
    fake_exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=_get_updated_graph_signature(
                real_exported_program.graph_signature,
                real_exported_program.graph_module,
            ),
        state_dict=new_state_dict,
        range_constraints=copy.deepcopy(real_exported_program.range_constraints),
        module_call_graph=copy.deepcopy(real_exported_program.module_call_graph),
        verifier=real_exported_program.verifier,
        constants=real_exported_program.constants,
    )
    return fake_exported_program


def update_to_real_program(
    fake_exported_program: ExportedProgram, real_exported_program: ExportedProgram
) -> None:
    """Update the fake exported program to point to the real state dict. Modifies the
    fake exported program in-place.
    """
    fake_exported_program._state_dict = real_exported_program.state_dict
