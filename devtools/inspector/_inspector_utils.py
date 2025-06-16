# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, IO, List, Mapping, Optional, Tuple, TypeAlias, Union

import executorch.devtools.etdump.schema_flatcc as flatcc

import pandas as pd

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

from tabulate import tabulate

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

EXCLUDED_EVENTS_FOR_INTERMEDIATE_OUTPUT = {"OPERATOR_CALL"}


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


class NodeSource(Enum):
    AOT = 1
    RUNTIME = 2


@dataclass
class NodeData:
    """
    Each node in the graph is an instance of NodeData, which contains:
    - source: A string indicating the origin of the node (either FROM_AOT or FROM_RUNTIME).
    - debug_handle: A tuple representing the unique identifier for the output.
    - output: The actual output data associated with the debug handle.
    """

    source: NodeSource
    debug_handle: tuple[int]
    output: Any


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
            ScalarType.BITS16: (torch.uint16, 2),
            ScalarType.UINT16: (torch.uint16, 2),
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
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `bool`.
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


def gen_etdump_object(
    etdump_path: Optional[str] = None, etdump_data: Optional[bytes] = None
) -> ETDumpFlatCC:
    # Gen event blocks from etdump
    if etdump_data is None and etdump_path is not None:
        with open(etdump_path, "rb") as buff:
            etdump_data = buff.read()

    if etdump_data is None:
        raise ValueError(
            "Unable to get ETDump data. One and only one of etdump_path and etdump_data must be specified."
        )

    return deserialize_from_etdump_flatcc(etdump_data)


def display_or_print_df(df: pd.DataFrame, file: IO[str] = sys.stdout):
    try:
        from IPython import get_ipython
        from IPython.display import display

        def style_text_size(val, size=12):
            return f"font-size: {size}px"

        if get_ipython() is not None:
            styled_df = df.style.applymap(style_text_size)
            display(styled_df)
        else:
            raise Exception(
                "Environment unable to support IPython. Fall back to print()."
            )
    except:
        print(
            tabulate(df, headers="keys", tablefmt="fancy_grid"),
            file=file,
        )


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
        return round((torch.pow((a - b), 2)).mean().item(), 2)

    results = []
    for ref_value, value in zip(ref_values, values):
        # TODO T171811011: extend the implementation of each metrics function to support value types other than tensor type
        if isinstance(ref_value, torch.Tensor) and isinstance(value, torch.Tensor):
            results.append(
                mean_squared_error(ref_value.to(torch.float32), value.to(torch.float32))
            )
        else:
            results.append(None)

    return results


def calculate_snr(ref_values: ProgramOutput, values: ProgramOutput):
    def signal_to_noise(signal: torch.Tensor, noise: torch.Tensor):
        signal_power = torch.mean(torch.pow(signal, 2))
        noise_power = torch.mean(torch.pow(noise, 2))
        snr = 10 * torch.log10(signal_power / noise_power)
        return round(snr.item(), 2)

    results = []
    for ref_value, value in zip(ref_values, values):
        # TODO T171811011: extend the implementation of each metrics function to support value types other than tensor type
        if isinstance(ref_value, torch.Tensor) and isinstance(value, torch.Tensor):
            ref_value_fp = ref_value.to(torch.float32)
            value_fp = value.to(torch.float32)
            diff = ref_value_fp - value_fp
            snr = signal_to_noise(ref_value_fp, diff)
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
            results.append(
                cosine_similarity(ref_value.to(torch.float32), value.to(torch.float32))
            )
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


def merge_overlapping_debug_handles(intermediate_outputs: Dict[Tuple[int, ...], Any]):
    """
    Merge overlapping debug handles int a single key
    """
    if len(intermediate_outputs) == 0:
        return
    # Extract and normalize into (start, end, val)
    intervals = [(min(key), max(key), val) for key, val in intermediate_outputs.items()]
    intervals.sort(key=lambda x: x[0])

    # Merge overlapping debug_hanldes, picking the last value
    merged_intermediate_outputs = []
    cur_start, cur_end, cur_val = intervals[0]
    for start, end, val in intervals[1:]:
        if start <= cur_end:  # Overlaps
            if end > cur_end:  # Extend if this one goes further
                cur_end, cur_val = end, val

        else:
            merged_intermediate_outputs.append((cur_start, cur_end, cur_val))
            cur_start, cur_end, cur_val = start, end, val
    merged_intermediate_outputs.append((cur_start, cur_end, cur_val))

    # Clear original one and populate with merged keys (value will point to the same object)
    intermediate_outputs.clear()
    for start, end, val in merged_intermediate_outputs:
        intermediate_outputs[tuple(range(start, end + 1))] = val


def _debug_handles_have_overlap(
    aot_debug_hanlde: Tuple[int, ...], runtime_debug_handle: Tuple[int, ...]
) -> bool:
    """
    Check if the AOT debug handle and the runtime debug handle have any overlap.
    """
    aot_set = set(aot_debug_hanlde)
    runtime_set = set(runtime_debug_handle)
    return len(aot_set.intersection(runtime_set)) > 0


def _combine_debug_hanldes(debug_handles: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Combine multiple debug handles into one debug handle"""
    combined_debug_handles_set = set()
    for debug_handle in debug_handles:
        combined_debug_handles_set.update(set(debug_handle))
    return tuple(sorted(combined_debug_handles_set))


def _combine_overlapped_intermediate_outputs(
    nodes: List[Tuple[Tuple[int, ...], Any]]
) -> Tuple[Tuple[int, ...], Any]:
    """Combine multiple overlapped intermediate outputs into one with combined debug_handles and last output"""
    debug_handles = [debug_handle for debug_handle, _ in nodes]
    outputs = [output for _, output in nodes]
    combined_debug_handle = _combine_debug_hanldes(debug_handles)
    output = outputs[-1]  # Pick the last one
    return combined_debug_handle, output


def _create_debug_handle_overlap_graph(
    aot_intermediate_outputs: Dict[Tuple[int, ...], Any],
    runtime_intermediate_outputs: Dict[Tuple[int, ...], Any],
) -> Tuple[List[NodeData], Dict[int, List[int]]]:
    """
    Create a graph representing overlapping debug handles between AOT and runtime outputs.

    Edges in the graph are represented as a dictionary where:
    - The key is the index of a node in the nodes list.
    - The value is a list of indices of nodes that have overlapping debug handles with the key node.

    Returns:
    - A tuple containing:
      - A list of NodeData instances representing the nodes in the graph.
      - A dictionary representing the edges, where each key-value pair indicates connected nodes due to overlapping debug handles.
    """
    nodes = []
    for debug_handle, output in aot_intermediate_outputs.items():
        nodes.append(NodeData(NodeSource.AOT, debug_handle, output))
    for debug_handle, output in runtime_intermediate_outputs.items():
        nodes.append(NodeData(NodeSource.RUNTIME, debug_handle, output))

    edges = {i: [] for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            # Only connect nodes from different sources(aot vs runtime) that overlap
            if node_i.source != node_j.source and _debug_handles_have_overlap(
                node_i.debug_handle, node_j.debug_handle
            ):
                edges[i].append(j)
                edges[j].append(i)
    return (nodes, edges)


def _find_connected_components(
    nodes: List[NodeData], edges: Dict[int, List[int]]
) -> List[List[int]]:
    """
    Find groups of connected nodes in a graph using DFS.
    Parameters:
    - nodes: A list of nodes in the graph.
    - edges: A dictionary where each key is a node index, and the value is a list
      of indices of connected nodes.
    Returns:
    - A list of connected components, each represented as a list of node indices.
    """
    visited = [False] * len(nodes)
    connected_components = []

    def dfs(node_id, component):
        visited[node_id] = True
        component.append(node_id)
        # Iterate over all neighbors of the current node
        for neighbor_node_id in edges[node_id]:
            # If a neighbor has not been visited yet, recursively visit it
            if not visited[neighbor_node_id]:
                dfs(neighbor_node_id, component)

    # Perform DFS on all nodes to find connected components
    for i in range(len(nodes)):
        # If a node has not been visited yet, start a new DFS from it
        if not visited[i]:
            component = []
            dfs(i, component)
            # After visiting all reachable nodes, add the current component to the list
            connected_components.append(component)
    return connected_components


def map_runtime_aot_intermediate_outputs(
    aot_intermediate_outputs: Dict[Tuple[int, ...], Any],
    runtime_intermediate_outputs: Dict[Tuple[int, ...], Any],
) -> Dict[Tuple[Tuple[int, ...], Any], Tuple[Tuple[int, ...], Any]]:
    """
    Map the runtime intermediate outputs to the AOT intermediate outputs
    by finding overlapping debug handles and combining them into a single debug_handle

    Returns:
        Dict[Tuple[Tuple[int, ...], Any], Tuple[Tuple[int, ...], Any]] - Mapping
        from runtime intermediate output to AOT intermediate output
    """
    # Merge overlapping debug handles
    merge_overlapping_debug_handles(aot_intermediate_outputs)
    merge_overlapping_debug_handles(runtime_intermediate_outputs)

    # Create a graph(nodes and edges) of overlapping(between aot and runtime) debug handles
    nodes, edges = _create_debug_handle_overlap_graph(
        aot_intermediate_outputs, runtime_intermediate_outputs
    )
    # Find connected(between aot and runtime) components
    connected_components = _find_connected_components(nodes, edges)

    aot_runtime_mapping = {}
    for comp in connected_components:
        # Separate nodes into AOT and runtime lists based on their source,
        # each list is combined into a single element and mapped to each other.
        aot_list = [
            (nodes[node_id].debug_handle, nodes[node_id].output)
            for node_id in comp
            if nodes[node_id].source == NodeSource.AOT
        ]
        runtime_list = [
            (nodes[node_id].debug_handle, nodes[node_id].output)
            for node_id in comp
            if nodes[node_id].source == NodeSource.RUNTIME
        ]

        # Map only if both AOT and runtime data are present.
        if len(aot_list) != 0 and len(runtime_list) != 0:
            # Combine aot debug handles into a single key
            aot_combined_debug_handle, aot_output = (
                _combine_overlapped_intermediate_outputs(aot_list)
            )
            # Combine runtime debug handles into a single key
            runtime_combined_debug_handle, runtime_output = (
                _combine_overlapped_intermediate_outputs(runtime_list)
            )
            # Create a mapping between runtime and aot
            aot_runtime_mapping[(aot_combined_debug_handle, aot_output)] = (
                runtime_combined_debug_handle,
                runtime_output,
            )

    return aot_runtime_mapping
