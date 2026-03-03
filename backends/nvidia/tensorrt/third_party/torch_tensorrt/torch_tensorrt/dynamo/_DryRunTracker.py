import logging
import math
import operator
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry
from torch_tensorrt.dynamo.conversion.converter_utils import get_node_name

logger = logging.getLogger(__name__)


@dataclass
class PerSubgraphData:
    """Class to track data on a per-subgraph level

    Args:
        subgraph_name (str): Name of the subgraph in the GraphModule
        subgraph_op_count (int): Number of operations in the subgraph
        input_shapes (Any): Shapes of input Tensors of the subgraph
        input_dtypes (Any): Input data types of the subgraph
        output_shapes (Any): Shapes of output Tensors of the subgraph
        output_dtypes (Any): Output data types of the subgraph
    """

    subgraph_name: str = ""
    subgraph_op_count: int = 0
    input_shapes: Any = field(default_factory=list)
    input_dtypes: Any = field(default_factory=list)
    output_shapes: Any = field(default_factory=list)
    output_dtypes: Any = field(default_factory=list)


@dataclass
class DryRunTracker:
    """Class to track data on a graph-wide level

    Args:
        total_ops_in_graph (int): Total number of operators in graph
        supported_ops_in_graph (int): Number of supported operators in graph
        input_shapes (Any): Shapes of input Tensors of the graph
        input_dtypes (Any): Input data types of the graph
        output_shapes (Any): Shapes of output Tensors of the graph
        output_dtypes (Any): Output data types of the graph
        per_subgraph_data (List[PerSubgraphData]): Per-subgraph data, see above class
        tensorrt_graph_count (int): Number of TensorRT engines to be generated
        compilation_settings (CompilationSettings): User Compilation Settings
        unsupported_ops (Dict[str, int]): Set of operators not supported in TRT
        to_run_in_torch (List[str]): Set of nodes to run in Torch
    """

    total_ops_in_graph: int = 0
    supported_ops_in_graph: int = 0
    input_shapes: Any = field(default_factory=list)
    input_dtypes: Any = field(default_factory=list)
    output_shapes: Any = field(default_factory=list)
    output_dtypes: Any = field(default_factory=list)
    per_subgraph_data: List[PerSubgraphData] = field(default_factory=list)
    tensorrt_graph_count: int = 0
    compilation_settings: CompilationSettings = field(
        default_factory=CompilationSettings
    )
    unsupported_ops: Dict[str, int] = field(default_factory=dict)
    to_run_in_torch: List[str] = field(default_factory=list)


def dryrun_stats_display(
    dryrun_tracker: DryRunTracker, dryrun_enabled: Union[bool, str]
) -> None:
    """Displays statistics about the dryrun either to debug logs or stdout"""
    formatted_stats = "\n"

    # Print overall stats about the graph, operator counts, etc.
    formatted_stats += "+" * 50 + " Dry-Run Results for Graph " + "+" * 50 + "\n\n"
    formatted_stats += (
        f"The graph consists of {dryrun_tracker.total_ops_in_graph} Total Operators, "
        f"of which {dryrun_tracker.supported_ops_in_graph} operators are supported, "
        f"{round(dryrun_tracker.supported_ops_in_graph*100/dryrun_tracker.total_ops_in_graph, 2)}% coverage\n\n"
    )
    if dryrun_tracker.unsupported_ops:
        parsed_ops = "\n".join(
            [f"{str(k)}: {str(v)}" for k, v in dryrun_tracker.unsupported_ops.items()]
        )
        formatted_stats += f"The following ops are currently unsupported or excluded from conversion, and are listed with their op-count in the graph:\n {parsed_ops}\n\n"

    if dryrun_tracker.to_run_in_torch:
        formatted_nodes = "\n".join(dryrun_tracker.to_run_in_torch)
        formatted_stats += (
            f"The following nodes are currently set to run in Torch:\n{formatted_nodes}\n"
            "Note: Some of the above nodes may be supported, but were not included in a TRT graph by the partitioner\n\n"
        )

    formatted_stats += f"Compiled with: {dryrun_tracker.compilation_settings}\n\n"

    assert len(dryrun_tracker.per_subgraph_data) == dryrun_tracker.tensorrt_graph_count

    # Print schematic of the graph structure, as in:
    #
    #   Inputs: List[Tensor: (1, 3, 224, 224)@float32]
    #    ...
    #      TRT Engine #1 - Submodule name: _run_on_acc_0
    #       Engine Inputs: List[Tensor: (1, 3, 224, 224)@float32]
    #       Number of Operators in Engine: 1
    #       Engine Outputs: Tensor: (1, 64, 112, 112)@float32
    #    ...
    #   Outputs: List[Tensor: (1, 1000)@float32]
    #
    formatted_stats += " " * 2 + "Graph Structure:\n\n"
    formatted_stats += (
        " " * 3
        + f"Inputs: {input_formatter(dryrun_tracker.input_shapes, dryrun_tracker.input_dtypes)}\n"
    )

    for i, trt_subgraph_data in enumerate(dryrun_tracker.per_subgraph_data):
        formatted_stats += " " * 4 + "...\n"
        formatted_stats += (
            " " * 4
            + f"TRT Engine #{i+1} - Submodule name: {trt_subgraph_data.subgraph_name}\n"
        )
        formatted_stats += (
            " " * 5
            + f"Engine Inputs: {input_formatter(trt_subgraph_data.input_shapes, trt_subgraph_data.input_dtypes)}\n"
        )
        formatted_stats += (
            " " * 5
            + f"Number of Operators in Engine: {trt_subgraph_data.subgraph_op_count}\n"
        )
        formatted_stats += (
            " " * 5
            + f"Engine Outputs: {input_formatter(trt_subgraph_data.output_shapes, trt_subgraph_data.output_dtypes)}\n"
        )

    formatted_stats += " " * 4 + "...\n"
    formatted_stats += (
        " " * 3
        + f"Outputs: {input_formatter(dryrun_tracker.output_shapes, dryrun_tracker.output_dtypes)}\n"
    )

    # Print aggregate statistics about the graph structure, including recommended "min_block_size" options
    if dryrun_tracker.tensorrt_graph_count > 0:
        min_ops_in_an_engine = min(
            trt_subgraph.subgraph_op_count
            for trt_subgraph in dryrun_tracker.per_subgraph_data
        )
        avg_ops_per_engine = (
            sum(
                trt_subgraph.subgraph_op_count
                for trt_subgraph in dryrun_tracker.per_subgraph_data
            )
            / dryrun_tracker.tensorrt_graph_count
        )
        avg_ops_per_engine = round(avg_ops_per_engine, 2)
        most_ops_in_an_engine = max(
            trt_subgraph.subgraph_op_count
            for trt_subgraph in dryrun_tracker.per_subgraph_data
        )

        formatted_stats += "\n" + " " * 2 + "-" * 25 + " Aggregate Stats " + "-" * 25
        formatted_stats += (
            "\n\n"
            + " " * 3
            + "Average Number of Operators per TRT Engine: "
            + f"{avg_ops_per_engine}"
        )

        formatted_stats += (
            "\n"
            + " " * 3
            + "Most Operators in a TRT Engine: "
            + f"{most_ops_in_an_engine}"
        )

        formatted_stats += "\n\n" + " " * 2 + "*" * 10 + " Recommendations " + "*" * 10
        formatted_stats += (
            "\n\n"
            + " " * 3
            + "- For minimal graph segmentation, select min_block_size="
            + f"{most_ops_in_an_engine} which would generate "
            + f"{len([1 for trt_subgraph in dryrun_tracker.per_subgraph_data if trt_subgraph.subgraph_op_count >= most_ops_in_an_engine])} TRT engine(s)"
        )
        if math.ceil(avg_ops_per_engine) != most_ops_in_an_engine:
            formatted_stats += (
                "\n"
                + " " * 3
                + "- For moderate graph segmentation, select min_block_size="
                + f"{math.ceil(avg_ops_per_engine)} which would generate "
                + f"{len([1 for trt_subgraph in dryrun_tracker.per_subgraph_data if trt_subgraph.subgraph_op_count >= math.ceil(avg_ops_per_engine)])} TRT engine(s)"
            )

        formatted_stats += (
            "\n"
            + " " * 3
            + "- The current level of graph segmentation is equivalent to selecting min_block_size="
            + f"{min_ops_in_an_engine} which generates "
            + f"{len([1 for trt_subgraph in dryrun_tracker.per_subgraph_data if trt_subgraph.subgraph_op_count >= min_ops_in_an_engine])} TRT engine(s)"
        )
    else:
        formatted_stats += (
            "\n"
            + " " * 2
            + "Aggregate stats not available since no TRT Engines were generated."
        )

    # If user specified "dryrun=True", print to stdout, else debug
    # If user specified a filepath, save the output to the path as well
    if dryrun_enabled:
        print(formatted_stats)
        if isinstance(dryrun_enabled, str):
            if os.path.exists(dryrun_enabled):
                logger.warning(
                    f"File already exists at path {dryrun_enabled}, not saving dryrun output"
                )
            else:
                with open(dryrun_enabled, "w+") as f:
                    f.write(formatted_stats)
    else:
        logger.debug(formatted_stats)


def input_formatter(shapes: Any, dtypes: Any) -> str:
    """Format shapes and dtypes of input Tensors into a readable string"""

    def input_formatter_helper(shapes: Any, dtypes: Any) -> str:
        """Helper for input formatter"""
        # Base case 1 - single static/dynamic shape, single dtype
        if isinstance(shapes, tuple) and all(
            isinstance(elt, (int, tuple)) for elt in shapes
        ):
            input_shape_string = "Tensor: ("
            for elt in shapes:
                if isinstance(elt, tuple):
                    input_shape_string += f"(min={elt[0]}, max={elt[1]}), "
                else:
                    input_shape_string += f"{elt}, "
            input_shape_string = input_shape_string[:-2] + ")" + f"@{str(dtypes)[6:]}, "
            return input_shape_string

        # Base case 2 - dynamic shape, single dtype
        elif (
            isinstance(shapes, dict)
            and len(shapes) == 3
            and all(
                (
                    isinstance(shape, tuple)
                    and all(isinstance(elt, int) for elt in shape)
                    and k in ("min_shape", "opt_shape", "max_shape")
                )
                for k, shape in shapes.items()
            )
        ):
            return f"Tensor: {shapes}@{str(dtypes)[6:]}, "

        # Shapes is a sequence
        elif isinstance(shapes, (list, tuple)):
            formatted_str = "List[" if isinstance(shapes, list) else "Tuple("
            for shape, dtype in zip(shapes, dtypes):
                formatted_str += input_formatter_helper(shape, dtype)
            formatted_str = formatted_str[:-2] + (
                "], " if isinstance(shapes, list) else "), "
            )
            return formatted_str

        # Shapes is a dictionary
        elif isinstance(shapes, dict):
            formatted_str = "Dict{"

            for key, shape in shapes.items():
                formatted_str += input_formatter_helper(shape, dtypes[key])

            formatted_str = formatted_str[:-2] + "}, "
            return formatted_str

        else:
            raise ValueError(
                f"Invalid input type {type(shapes)} encountered in parse_complex_tensor_structs parsing."
            )

    return input_formatter_helper(shapes, dtypes)[:-2]


def parse_non_trt_nodes(graph_module: torch.fx.GraphModule) -> List[str]:
    """Parses call_function and call_method nodes from a GraphModule
    Excludes getitem nodes

    Returns a string representation of the nodes
    """
    to_run_in_torch = []
    for node in graph_module.graph.nodes:
        # getitem nodes are excluded since they are a Tensor-collection op
        if (
            node.op in ("call_function", "call_method")
            and node.target != operator.getitem
        ):
            to_run_in_torch.append(
                f"Node: {ConverterRegistry.qualified_name_or_str(node.target)}, "
                f"with layer location: {get_node_name(node)}"
            )
    return to_run_in_torch
