# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import executorch.extension.pytree as ex_pytree
import torch
import torch.fx
from executorch.exir.emit._emitter import (
    _DelegateDebugIdentifierMap,
    _EmitterState,
    _ProgramState,
    _TopLevelEmitter,
)
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.schema import (
    Bool,
    Chain,
    ContainerMetadata,
    Double,
    EValue,
    ExecutionPlan,
    Int,
    Program,
    String,
)
from executorch.exir.tensor import layout_enum, scalar_type_enum
from executorch.exir.version import EXECUTORCH_SCHEMA_VERSION
from torch._export.exported_program import ExportedProgram


def _emit_prim_getters(prim_getters: Dict[str, Any]) -> List[ExecutionPlan]:
    """
    Given a mapping of function names to return values, emit simple execution
    plans that just return these constant values.

    Precondition: All the values are primitives (bool, float, int, str, enum)
    or structures (list, dict) of them.
    """
    plans = []
    # flatten any structures
    for method, vals in prim_getters.items():
        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        flattened_output, spec = ex_pytree.tree_flatten(vals)
        spec = spec.to_str()
        chain = Chain(
            inputs=[],
            outputs=[],
            instructions=[],
            stacktrace=None,
        )

        # switch on type of prim
        values = []
        for val in flattened_output:
            if isinstance(val, float):
                values.append(EValue(Double(val)))

            elif isinstance(val, bool):
                values.append(EValue(Bool(val)))

            elif isinstance(val, int):
                values.append(EValue(Int(val)))

            elif isinstance(val, str):
                values.append(EValue(String(val)))

            elif isinstance(val, torch.dtype):
                values.append(EValue(Int(scalar_type_enum(val))))

            elif isinstance(val, torch.layout):
                values.append(EValue(Int(layout_enum(val))))

            else:
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"Error emitting {method} which returns a value of type {type(val)}. which is not a supported primitive",
                )

        # add to plans
        plans.append(
            ExecutionPlan(
                name=method,
                values=values,
                inputs=[],
                outputs=list(range(0, len(values))),
                chains=[chain],
                operators=[],
                delegates=[],
                non_const_buffer_sizes=[0, 0],
                container_meta_type=ContainerMetadata("", spec),
            )
        )
    return plans


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
        if (
            exported_program.graph_signature.buffers_to_mutate
        ):  # see if we are mutating any state
            raise ExportError(
                ExportErrorType.INVALID_INPUT_TYPE,
                "Buffers cannot be modified in executorch.",
            )
        # create empty state
        emitter_state = _EmitterState(
            values=[],
            operators=[],
            delegates=[],
            num_values=0,
            operator_cache={},
            delegate_cache={},
            emit_stacktrace=emit_stacktrace,
        )

        emitter = _TopLevelEmitter(name, exported_program, program_state, emitter_state)

        emitter.run()
        plans.append(emitter.plan())

        debug_handle_map[name] = emitter.debug_handle_map
        method_to_delegate_debug_id_map[
            name
        ] = emitter.instr_id_to_delegate_debug_id_map

    # emit any primitive getters
    if prim_getters is not None:
        plans.extend(_emit_prim_getters(prim_getters))

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
        ),
    )
