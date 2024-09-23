# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple, TypeAlias, Union

import executorch.devtools.etdump.schema_flatcc as flatcc

import torch

from executorch.devtools.debug_format.base_schema import OperatorNode

from executorch.devtools.debug_format.et_schema import FXOperatorGraph, OperatorGraph
from executorch.devtools.etdump.schema_flatcc import (
    DebugEvent,
    ETDumpFlatCC,
    ProfileEvent,
    ScalarType,
    Tensor,
    Value,
    ValueType,
)

from executorch.devtools.etdump.serialize import deserialize_from_etdump_flatcc
from executorch.devtools.etrecord import ETRecord

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


def calculate_time_scale_factor(
    source_time_scale: TimeScale, target_time_scale: TimeScale
) -> float:
    """
    Calculate the factor (source divided by target) between two time scales
    """
    return TIME_SCALE_DICT[source_time_scale] / TIME_SCALE_DICT[target_time_scale]


# Model Debug Output
InferenceOutput: TypeAlias = Union[
    torch.Tensor, List[torch.Tensor], int, float, str, bool, None
]
ProgramOutput: TypeAlias = List[InferenceOutput]


# Compare whether two InferenceOutputs are equal
def is_inference_output_equal(
    output1: InferenceOutput, output2: InferenceOutput
) -> bool:
    if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return torch.equal(output1, output2)
    elif isinstance(output1, List) and isinstance(output2, List):
        return all(torch.equal(t1, t2) for t1, t2 in zip(output1, output2))
    elif output1 == output2:
        return True
    else:
        return False


# Given a ETDump Tensor object and offset, extract into a torch.Tensor
def _parse_tensor_value(
    tensor: Optional[Tensor], output_buffer: Optional[bytes]
) -> torch.Tensor:
    def get_scalar_type_size(scalar_type: ScalarType) -> Tuple[torch.dtype, int]:
        """
        Return the size of the scalar type in bytes
        """
        get_scalar_type_size_map = {
            ScalarType.BYTE: (torch.uint8, 1),
            ScalarType.CHAR: (torch.int8, 1),
            ScalarType.BOOL: (torch.bool, 1),
            ScalarType.SHORT: (torch.int16, 2),
            ScalarType.HALF: (torch.float16, 2),
            ScalarType.INT: (torch.int, 4),
            ScalarType.FLOAT: (torch.float, 4),
            ScalarType.DOUBLE: (torch.double, 8),
            ScalarType.LONG: (torch.long, 8),
        }
        if scalar_type in get_scalar_type_size_map:
            return get_scalar_type_size_map[scalar_type]
        else:
            raise RuntimeError(
                f"Unsupported scalar type in get_scalar_type_size : {scalar_type}"
            )

    if tensor is None or tensor.offset is None:
        raise ValueError("Tensor cannot be None")

    torch_dtype, dtype_size = get_scalar_type_size(tensor.scalar_type)

    if output_buffer is None:
        # Empty buffer provided. Cannot deserialize tensors.
        return torch.zeros(tensor.sizes, dtype=torch_dtype)

    tensor_bytes_size = math.prod(tensor.sizes) * dtype_size
    if tensor_bytes_size == 0:
        # Empty tensor. Return empty tensor.
        return torch.zeros(tensor.sizes, dtype=torch_dtype)

    if tensor.offset is None:
        raise ValueError("Tensor offset cannot be None")

    return torch.frombuffer(
        output_buffer[tensor.offset : tensor.offset + tensor_bytes_size],
        dtype=torch_dtype,
    ).view(tensor.sizes)


def inflate_runtime_output(
    value: Value, output_buffer: Optional[bytes]
) -> InferenceOutput:
    """
    Parse the given ETDump Value object into an InferenceOutput object
    """

    if value.val == ValueType.INT.value:
        if value.int_value is None:
            raise ValueError("Expected Int value, `None` provided")
        return value.int_value.int_val
    if value.val == ValueType.BOOL.value:
        if value.bool_value is None:
            raise ValueError("Expected Bool value, `None` provided")
        return value.bool_value.bool_val
    if value.val == ValueType.FLOAT.value:
        if value.float_value is None:
            raise ValueError("Expected Float value, `None` provided")
        return value.float_value.float_val
    if value.val == ValueType.DOUBLE.value:
        if value.double_value is None:
            raise ValueError("Expected Double value, `None` provided")
        return value.double_value.double_val
    if value.val == ValueType.TENSOR.value:
        return _parse_tensor_value(value.tensor, output_buffer)
    if value.val == ValueType.TENSOR_LIST.value:
        if value.tensor_list is None:
            raise ValueError("Expected TensorList value, `None` provided")
        return [
            _parse_tensor_value(t, output_buffer) for t in value.tensor_list.tensors
        ]


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

    for output_a, output_b in zip(existing_data, new_data):
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
    etrecord: ETRecord, enable_module_hierarchy: bool = False
) -> Mapping[str, OperatorGraph]:
    op_graph_map = {}
    if etrecord.graph_map is not None:
        op_graph_map = {
            name: FXOperatorGraph.gen_operator_graph(
                exported_program.graph_module,
                enable_module_hierarchy=enable_module_hierarchy,
            )
            for name, exported_program in etrecord.graph_map.items()
        }
    if etrecord.edge_dialect_program is not None:
        op_graph_map[EDGE_DIALECT_GRAPH_KEY] = FXOperatorGraph.gen_operator_graph(
            etrecord.edge_dialect_program.graph_module,
            enable_module_hierarchy=enable_module_hierarchy,
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


def plot_metric(result: List[float], metric_name: str):
    import matplotlib.pyplot as plt
    import numpy as np

    # Clear the current figure, otherwise this plot will be on top of previous plots
    plt.clf()
    plt.figure(figsize=(8, 6))

    x_axis = np.arange(len(result))
    bars = plt.bar(x_axis, result, width=0.5)
    plt.grid(True, which="major", axis="y")
    num_ticks = len(x_axis) if len(x_axis) > 5 else 5
    interval = 1 if num_ticks < 20 else 5
    plt.xticks(list(range(num_ticks))[::interval])
    plt.xlabel("Output value index")
    plt.ylabel(metric_name)
    plt.title(f"Metric {metric_name}")

    # Add value annotations to each bar
    for bar, value in zip(bars, result):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    max_value = max(result) * 1.25
    min_value = min(result) * 1.25

    # Cosine similarity has range [-1, 1], so we set y-axis limits accordingly.
    if metric_name == "cosine_similarity":
        max_value = 1.0
        if min_value >= 0:
            min_value = 0
        else:
            min_value = -1.0

    plt.ylim(min(0, min_value), max(0, max_value))

    plt.savefig(f"{metric_name}_output_plot.png")  # Save the plot to a file
    plt.show()


def calculate_mse(ref_values: ProgramOutput, values: ProgramOutput):
    def mean_squared_error(a: torch.Tensor, b: torch.Tensor):
        return round((torch.pow((a - b).to(torch.float32), 2)).mean().item(), 2)

    results = []
    for ref_value, value in zip(ref_values, values):
        # TODO T171811011: extend the implementation of each metrics function to support value types other than tensor type
        if isinstance(ref_value, torch.Tensor) and isinstance(value, torch.Tensor):
            results.append(mean_squared_error(ref_value, value))
        else:
            results.append(None)

    return results


def calculate_snr(ref_values: ProgramOutput, values: ProgramOutput):
    def signal_to_noise(signal: torch.Tensor, noise: torch.Tensor):
        signal = signal.type(torch.float32)
        noise = noise.type(torch.float32)
        signal_power = torch.mean(torch.pow(signal, 2))
        noise_power = torch.mean(torch.pow(noise, 2))
        snr = 10 * torch.log10(signal_power / noise_power)
        return round(snr.item(), 2)

    results = []
    for ref_value, value in zip(ref_values, values):
        # TODO T171811011: extend the implementation of each metrics function to support value types other than tensor type
        if isinstance(ref_value, torch.Tensor) and isinstance(value, torch.Tensor):
            diff = ref_value - value
            snr = signal_to_noise(ref_value, diff)
            results.append(snr)
        else:
            results.append(None)

    return results


def calculate_cosine_similarity(ref_values: ProgramOutput, values: ProgramOutput):
    def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor):
        # Ensure that the tensors have the same shape
        if tensor1.shape != tensor2.shape:
            raise ValueError("Input tensors must have the same shape")

        # Calculate the dot product
        dot_product = torch.sum(tensor1 * tensor2)

        # Calculate the magnitudes
        magnitude1 = torch.sqrt(torch.sum(torch.pow(tensor1, 2)))
        magnitude2 = torch.sqrt(torch.sum(torch.pow(tensor2, 2)))

        # Calculate the cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        return round(similarity.item(), 2)  # Convert the result to a Python float

    results = []
    for ref_value, value in zip(ref_values, values):
        # TODO T171811011: extend the implementation of each metrics function to support value types other than tensor type
        if isinstance(ref_value, torch.Tensor) and isinstance(value, torch.Tensor):
            results.append(cosine_similarity(ref_value, value))
        else:
            results.append(None)

    return results


def compare_results(
    reference_output: ProgramOutput,
    run_output: ProgramOutput,
    metrics: Optional[List[str]] = None,
    plot: bool = False,
) -> Dict[str, List[float]]:
    """
    Compares the results of two runs and returns a dictionary of metric names -> lists of metric values. This list matches
    the reference output & run output lists, so essentially we compare each pair of values in those two lists.

    Args:
        reference_output: Reference program output.
        run_output: Program output to compare with reference output.
        metrics: List of requested metric names. Defaults to all available metrics.
        plot: Whether to plot the results.

    Returns:
        Dictionary of metric names to lists of float values.
    """

    results = {}
    metrics_functions = {
        "snr": calculate_snr,
        "mse": calculate_mse,
        "cosine_similarity": calculate_cosine_similarity,
    }
    for supported_metric in metrics_functions:
        if metrics is None or supported_metric in metrics:
            result = metrics_functions[supported_metric](reference_output, run_output)
            results[supported_metric] = result

            if plot:
                plot_metric(result, supported_metric)
            else:
                print(supported_metric)
                print("-" * 20)
                for index, value in enumerate(result):
                    print(f"{index:<5}{value:>8.5f}")
                print("\n")

    return results
