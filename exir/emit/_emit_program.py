# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.fx
from executorch.exir.emit._emitter import (
    _DelegateDebugIdentifierMap,
    _EmitterState,
    _ProgramState,
    _TopLevelEmitter,
)
from executorch.exir.error import ExportError, ExportErrorType

from executorch.exir.schema import Buffer, Program, SubsegmentOffsets
from executorch.exir.version import EXECUTORCH_SCHEMA_VERSION
from torch.export.exported_program import ExportedProgram, OutputKind
from torch.utils import _pytree as pytree


@dataclass
class EmitterOutput:
    """
    The outputs of program emission. Contains the executorch program object as well as
    a mapping of instruction ids to debug handles.
    """

    # The ExecuTorch program
    program: Program

    # This dictionary maps the instruction ids to their corresponding
    # debug handles or list of debug handles in the case of delegate calls.
    debug_handle_map: Dict[int, Union[int, List[int]]]

    # This dictionary maps the method name to the corresponding dict which
    # contains the mapping of the delegate instruction id to its corresponding
    # delegate name and delegate debug identifier mapping.
    method_to_delegate_debug_id_map: Dict[
        str, Dict[int, Dict[str, Union[str, _DelegateDebugIdentifierMap]]]
    ]

    mutable_data: Optional[List[Buffer]]


def _remove_non_user_outputs(exported_program: ExportedProgram) -> torch.fx.GraphModule:
    gm = exported_program.graph_module
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
    assert output_node is not None

    mutated_outputs: List[Optional[str]] = [
        out_spec.target if out_spec.kind in (OutputKind.BUFFER_MUTATION,) else None
        for out_spec in exported_program.graph_signature.output_specs
    ]
    outputs = pytree.tree_flatten(output_node.args)[0]

    user_output_nodes = []
    for return_node, mutated_node_name in zip(outputs, mutated_outputs):
        if mutated_node_name is None:
            user_output_nodes.append(return_node)
            continue

    with gm.graph.inserting_before(output_node):
        # Only return user outputs
        new_output = gm.graph.output(tuple(user_output_nodes))
        new_output.meta = output_node.meta.copy()
        output_node.replace_all_uses_with(new_output)
        gm.graph.erase_node(output_node)

    return gm


# For each entry point in the model, determine if its a joint graph,
# and if it is return a map of the indices in the model output that the
# gradient outputs start at and that the parameter outputs start at.
def _get_training_metadata(methods: Dict[str, ExportedProgram]) -> Dict[str, int]:
    gradients_method_prefix = "__et_training_gradients_index_"
    parameters_method_prefix = "__et_training_parameters_index_"
    fqn_method_prefix = "__et_training_fqn_"
    training_metadata = {}
    for name, method in methods.items():
        found_grad = False
        found_param = False
        fqns = []
        i = 0
        for output_spec in method.graph_signature.output_specs:
            if output_spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
                if not found_grad:
                    training_metadata[gradients_method_prefix + name] = i
                    found_grad = True
                fqns.append(output_spec.target)
            elif output_spec.kind == OutputKind.TOKEN and not found_param:
                assert found_grad  # Params must come after gradients
                training_metadata[parameters_method_prefix + name] = i
                found_param = True
            i += 1
            if len(fqns) > 0:
                training_metadata[fqn_method_prefix + name] = fqns
    return training_metadata


def emit_program(
    methods: Union[ExportedProgram, Dict[str, ExportedProgram]],
    emit_stacktrace: bool = False,
    prim_getters: Optional[Dict[str, Any]] = None,
) -> EmitterOutput:
    """
    Given a exported program, it returns the program in the format
    of the Python version of the flatbuffer Program schema.

    Args:
        methods: Either the exported program (Exported_Program) that we want to
            emit into the flatbuffer, or a dictionary of method names to
            ExportedPrograms.
        emit_stacktrace: Flag to enable emission of a stacktrace for each
           instruction for debugging purposes

    Return:
        The program in a Python class which mimics the flatbuffer schema
    """

    if isinstance(methods, ExportedProgram):
        methods = {"forward": methods}

    # validation
    bad_methods = []
    for name, exported_program in methods.items():
        if not isinstance(exported_program, ExportedProgram):
            bad_methods.append(name)
    if len(bad_methods) != 0:
        raise ExportError(
            ExportErrorType.INVALID_INPUT_TYPE,
            f"Did not receive ExportedProgram for the following methods {str(bad_methods)}",
        )

    plans = []
    debug_handle_map = {}
    method_to_delegate_debug_id_map = {}
    program_state = _ProgramState()

    # emit each entry point in order according to name.
    for name, exported_program in sorted(methods.items()):
        # create empty state
        emitter_state = _EmitterState(
            values=[],
            operators=[],
            delegates=[],
            operator_cache={},
            delegate_cache={},
            emit_stacktrace=emit_stacktrace,
        )

        gm = _remove_non_user_outputs(exported_program)

        emitter = _TopLevelEmitter(
            name, exported_program, gm, program_state, emitter_state
        )

        emitter.run()
        plans.append(emitter.plan())

        debug_handle_map[name] = emitter.debug_handle_map
        method_to_delegate_debug_id_map[name] = (
            emitter.instr_id_to_delegate_debug_id_map
        )

    training_metadata = _get_training_metadata(methods)
    if len(training_metadata) > 0:
        plans.extend(emitter._emit_prim_getters(training_metadata))

    # emit any primitive getters
    if prim_getters is not None:
        plans.extend(emitter._emit_prim_getters(prim_getters))

    return EmitterOutput(
        debug_handle_map=debug_handle_map,
        method_to_delegate_debug_id_map=method_to_delegate_debug_id_map,
        program=Program(
            version=EXECUTORCH_SCHEMA_VERSION,
            execution_plan=plans,
            constant_buffer=program_state.constant_buffer,
            backend_delegate_data=program_state.backend_delegate_data,
            # Segments may be added at serialization time.
            segments=[],
            # Subsegment offsets may be added at serialization time.
            constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
            mutable_data_segments=None,  # Will be filled in during serialization
        ),
        mutable_data=(
            program_state.mutable_buffer
            if len(program_state.mutable_buffer) > 1
            else None
        ),
    )
