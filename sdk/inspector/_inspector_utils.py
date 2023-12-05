# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple, TypeAlias, Union

import executorch.sdk.etdump.schema_flatcc as flatcc

import torch

from executorch.sdk.debug_format.base_schema import OperatorNode

from executorch.sdk.debug_format.et_schema import FXOperatorGraph, OperatorGraph
from executorch.sdk.etdump.schema_flatcc import (
    DebugEvent,
    ETDumpFlatCC,
    ProfileEvent,
    ScalarType,
    Tensor,
    Value,
    ValueType,
)

from executorch.sdk.etdump.serialize import deserialize_from_etdump_flatcc
from executorch.sdk.etrecord import ETRecord

FORWARD = "forward"
EDGE_DIALECT_GRAPH_KEY = "edge_dialect_graph_module"

RESERVED_FRAMEWORK_EVENT_NAMES = [
    "Method::init",
    "Program::load_method",
    "Method::execute",
]
EXCLUDED_COLUMNS_WHEN_PRINTING = [
    "raw",
    "delegate_debug_identifier",
    "stack_traces",
    "module_hierarchy",
    "debug_data",
]
EXCLUDED_EVENTS_WHEN_PRINTING = {"OPERATOR_CALL"}


class TimeScale(Enum):
    NS = "ns"
    US = "us"
    MS = "ms"
    S = "s"
    CYCLES = "cycles"


TIME_SCALE_DICT = {
    TimeScale.NS: 1000000000,
    TimeScale.US: 1000000,
    TimeScale.MS: 1000,
    TimeScale.S: 1,
    TimeScale.CYCLES: 1,
}


# Model Debug Output
InferenceOutput: TypeAlias = Union[torch.Tensor, int, float, str, bool, None]
ProgramOutput: TypeAlias = List[InferenceOutput]


def inflate_runtime_output(
    value: Value, output_buffer: Optional[bytes]
) -> InferenceOutput:
    """
    Parse the given ETDump Value object into an InferenceOutput object
    """

    def get_scalar_type_size(scalar_type: ScalarType) -> Tuple[torch.dtype, int]:
        """
        Return the size of the scalar type in bytes
        """
        match scalar_type:
            case ScalarType.INT:
                return (torch.int, 4)
            case ScalarType.BOOL:
                return (torch.bool, 1)
            case ScalarType.FLOAT:
                return (torch.float, 4)
            case ScalarType.DOUBLE:
                return (torch.double, 8)
            case ScalarType.LONG:
                return (torch.long, 8)
            case _:
                raise RuntimeError(
                    f"Unsupported scalar type in get_scalar_type_size : {scalar_type}"
                )

    # Given a ETDump Tensor object and offset, extract into a torch.Tensor
    def parse_tensor_value(tensor: Optional[Tensor]) -> torch.Tensor:
        if output_buffer is None:
            raise ValueError("Empty buffer provided. Cannot deserialize tensors.")
        if tensor is None or tensor.offset is None:
            raise ValueError("Tensor cannot be None")

        torch_dtype, dtype_size = get_scalar_type_size(tensor.scalar_type)
        tensor_bytes_size = math.prod(tensor.sizes) * dtype_size

        if tensor.offset is None:
            raise ValueError("Tensor offset cannot be None")

        return torch.frombuffer(
            output_buffer[tensor.offset : tensor.offset + tensor_bytes_size],
            dtype=torch_dtype,
        ).view(tensor.sizes)

    match value.val:
        case ValueType.INT.value:
            if value.int_value is None:
                raise ValueError("Expected Int value, `None` provided")
            return value.int_value.int_val
        case ValueType.BOOL.value:
            if value.bool_value is None:
                raise ValueError("Expected Bool value, `None` provided")
            return value.bool_value.bool_val
        case ValueType.FLOAT.value:
            if value.float_value is None:
                raise ValueError("Expected Float value, `None` provided")
            return value.float_value.float_val
        case ValueType.DOUBLE.value:
            if value.double_value is None:
                raise ValueError("Expected Double value, `None` provided")
            return value.double_value.double_val
        case ValueType.TENSOR.value:
            return parse_tensor_value(value.tensor)


def find_populated_event(event: flatcc.Event) -> Union[ProfileEvent, DebugEvent]:
    """
    Given a ETDump Event object, find the populated event

    Raise an error if no populated event can be found
    """
    if event.profile_event is not None:
        return event.profile_event

    if event.debug_event is not None:
        return event.debug_event

    raise ValueError("Unable to find populated event")


# TODO: Optimize by verifying prior to inflating the tensors
def verify_debug_data_equivalence(
    existing_data: ProgramOutput, new_data: ProgramOutput
) -> None:
    """
    Verify that the lists of inference_outputs are equivalent

    Raises an corresponding errors if they are not
    """
    assert len(existing_data) == len(
        new_data
    ), "Unequal debug data length encountered. Expected to be equal."

    for (output_a, output_b) in zip(existing_data, new_data):
        assert isinstance(
            output_a, type(output_b)
        ), "Debug Data Types are different. Expected to be equal."

        if isinstance(output_a, torch.Tensor):
            assert bool(
                torch.all(output_a == output_b)
            ), "Tensors Debug Data is different. Expected to be equal."
        else:
            assert (
                output_a == output_b
            ), "Scalar Debug Data is different. Expected to be equal"


def is_debug_output(value: Value) -> bool:
    """
    Returns True if the given flatcc.Value is a debug output
    """
    return value.output is not None and value.output.bool_val


def gen_graphs_from_etrecord(
    etrecord: ETRecord,
) -> Mapping[str, OperatorGraph]:
    op_graph_map = {}
    if etrecord.graph_map is not None:
        op_graph_map = {
            name: FXOperatorGraph.gen_operator_graph(exported_program.graph_module)
            for name, exported_program in etrecord.graph_map.items()
        }
    if etrecord.edge_dialect_program is not None:
        op_graph_map[EDGE_DIALECT_GRAPH_KEY] = FXOperatorGraph.gen_operator_graph(
            etrecord.edge_dialect_program.graph_module
        )

    return op_graph_map


def create_debug_handle_to_op_node_mapping(
    op_graph: OperatorGraph,
) -> Dict[int, OperatorNode]:
    """
    Recursive function to traverse all the operator graph nodes of input op_graph and build a mapping
    from each debug handle to the operator node that contains the debug handle in its metadata.
    """
    debug_handle_to_op_node_map: Dict[int, OperatorNode] = {}

    # Recursively searches through the metadata of nodes
    def _extract_debug_handles(graph: OperatorGraph):
        for element in graph.elements:
            if isinstance(element, OperatorGraph):
                _extract_debug_handles(element)
            if isinstance(element, OperatorNode) and element.metadata is not None:
                metadata = element.metadata
                debug_handle = metadata.get("debug_handle")
                if debug_handle is not None:
                    existing_entry = debug_handle_to_op_node_map.get(debug_handle)
                    if existing_entry is not None:
                        raise ValueError(
                            f"Duplicated debug handle {str(debug_handle)} shared between {element.name} and {existing_entry.name}. "
                            "No two op nodes of the same graph should have the same debug handle."
                        )
                    debug_handle_to_op_node_map[debug_handle] = element

    # Start traversing
    _extract_debug_handles(op_graph)
    return debug_handle_to_op_node_map


def gen_etdump_object(etdump_path: Optional[str] = None) -> ETDumpFlatCC:
    # Gen event blocks from etdump
    if etdump_path is None:
        raise ValueError("Etdump_path must be specified.")
    with open(etdump_path, "rb") as buff:
        etdump = deserialize_from_etdump_flatcc(buff.read())
        return etdump
