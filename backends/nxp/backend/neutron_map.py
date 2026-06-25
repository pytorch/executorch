# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import re
from dataclasses import dataclass

# example:  Type: CONV_2D
#               Inputs:
#                 [0]: quantized_decomposed_quantize_per_tensor_default_4
#                 [1]: quantized_decomposed_dequantize_per_channel_default_2
#               Outputs:
#                 [0]: quantized_decomposed_quantize_per_tensor_default_5
#               Location: 4
PATTERN_NODE = (
    r"Type:\s+(?P<type>\w+)\s+"
    r"Inputs:(?P<inputs>[\s\S]*?)"
    r"Outputs:(?P<outputs>[\s\S]*?)"
    r"Location:\s+(?P<location>\d+)"
)
# The pattern is very similar to operator pattern
PATTERN_SUBGRAPH = (
    r"^(?P<num>\d+)\s*"
    r"Inputs:(?P<inputs>[\s\S]*?)"
    r"Outputs:(?P<outputs>[\s\S]*?)"
    r"Tensors:"
)
# example:  [0]: quantized_decomposed_quantize_per_tensor_default_4
PATTERN_IO_TENSOR_NAME = r"\[\d+\]:\s+(?P<name>[\S]+)"
# example: Statistics for NeutronGraph "subgraph_195":
PATTERN_GRAPH = r"Statistics for NeutronGraph \"subgraph_(?P<num>\d+)\":"
# example:      NeutronOperator "subgraph_001":
#                       Operators:
#                           PAD
#                           CONV_2D
#                       Kernels:
#                           Pad
#                           Conv2DStandardV2
#               NeutronOperator "subgraph_002":
PATTERN_VERBOSE_KERNELS = (
    r"\"subgraph_(?P<subgraph>\d+)\"\:\s*"
    r"Operators:[\s\S]*?"
    r"Kernels:\s*(?P<kernels>[\s\S]*?)"
    r"\s*(NeutronOperator|^$|=)"
)
# example:  NeutronGraph "subgraph_074":
PATTERN_VERBOSE_GRAPH = (
    r"NeutronGraph\s*\"subgraph_(?P<subgraph>\d+)\":(?P<operators>[\s\S]*?)\s*(^$|=)"
)
# Two graphs are expected in the input log: original and converted.
EXPECTED_GRAPHS = 2
# List of single-input nodes that shouldn't be mapped on the same TFLite node.
SINGLE_INPUT_NODES = [
    "ABS",
    "AVERAGE_POOL_2D",
    "CAST",
    "EXP",
    "HARD_SWISH",
    "LEAKY_RELU",
    "LOG",
    "LOGISTIC",
    "MAX_POOL_2D",
    "QUANTIZE",
    "RSQRT",
    "TANH",
]


@dataclass
class Node:
    name: str  # Name of the node.
    inputs: list[str]  # List of nodes inputs.
    outputs: list[str]  # List of nodes outputs.
    location: int  # Location in graph/subgraph.


@dataclass
class SubgraphInfo:
    num: int  # Subgraph number.
    location: int  # Location in neutron graph
    inputs: list[str]  # List of subgraphs inputs.
    outputs: list[str]  # List of subgraphs outputs.
    kernels: int  # Number of neutron kernels in neutron subgraph.
    nodes: list[Node]  # List of tflite nodes in neutron subgraph.


def get_tensors_name(tensors: str) -> list[str]:
    """Split input string with tensor names into list of names"""
    return [m.group("name") for m in re.finditer(PATTERN_IO_TENSOR_NAME, tensors)]


class NeutronMap:
    """Mapping between Neutron, TFLite, and Edge operators based on the Neutron converter log.

    Parses the Neutron converter log to extract information about TFLite nodes and Neutron subgraphs.
    Maps TFLite operators to corresponding Neutron operators.
    Maps Edge operators to Neutron operators via the Edge-to-TFLite mapping.

    Attributes:
        tflite_nodes (list[Node]): TFLite node information extracted from the converter log.
        neutron_subgraphs (list[SubgraphInfo]): Neutron subgraph information extracted from the converter log.
        neutron_graphs (list[int]): Indices of final Neutron graphs derived from neutron_subgraphs.
        edge_to_tflite_map (dict[int, tuple[int, ...]]): Mapping from Edge operators to TFLite operators.
        edge_to_neutron_map (dict[int, tuple[int, ...]]): Mapping from Edge operators to Neutron operators.
        tflite_to_neutron_map (dict[int, tuple[int, ...]]): Mapping from TFLite operators to Neutron operators.

    Example:
        >>> map = NeutronMap(log_output, edge_to_tflite_map)
        >>> neutron_to_edge_map = map.get_neutron_to_edge_map()
    """

    tflite_nodes: list[Node]
    neutron_subgraphs: list[SubgraphInfo]
    neutron_graphs: list[int]
    edge_to_tflite_map: dict[int, tuple[int, ...]]
    edge_to_neutron_map: dict[int, tuple[int, ...]]
    tflite_to_neutron_map: dict[int, tuple[int, ...]]

    def __init__(
        self, neutron_converter_log: str, edge_to_tflite_map: dict[int, tuple[int, ...]]
    ) -> None:
        """Initialize neutron map from neutron converter log.

        :param neutron_converter_log: neutron converter log obtained during model conversion. It should contain
        original tflite graph and neutron graph dump. To add these dumps to converter log the dumpAfterImport and
        dumpAfterGenerate flags have to be set to "console".
        """
        super().__init__()
        self.tflite_nodes = []
        self.neutron_subgraphs = []
        self.neutron_graphs = []
        self.edge_to_tflite_map = edge_to_tflite_map
        self.tflite_to_neutron_map = {}
        self.edge_to_neutron_map = {}
        self.neutron_kernels_num = 0
        self._split_profiling_log(neutron_converter_log)

    def _split_profiling_log(self, log: str) -> None:
        """Process profiling log to split it into original TFLite and converted Neutron nodes.

        :param log: Neutron converter log obtained during model conversion, containing the original
            TFLite graph and Neutron graph dump.
        :return: None. Sets class attributes tflite_nodes and neutron_subgraphs with node information.
        """
        graphs = log.split("Graphs:")
        # Check if there is two graphs in the input dump
        if len(graphs) != EXPECTED_GRAPHS + 1:
            return
        optimization_dump, neutron_graph_dump = graphs[1:]

        # Get tflite model dump
        tflite_graph_dump = optimization_dump.partition("= Optimize Graph =")[0]

        # Get verbose Neutron graphs located in the Extract Graphs section.
        extracted_graph_dump = optimization_dump.partition("= Extract Graphs =")[
            2
        ].partition("Generate code for NeutronGraph")[0]

        # Get list of original operators from first dumped graph.
        self.tflite_nodes = [
            Node(
                matched_operator.group("type"),
                get_tensors_name(matched_operator.group("inputs")),
                get_tensors_name(matched_operator.group("outputs")),
                int(matched_operator.group("location")),
            )
            for matched_operator in re.finditer(PATTERN_NODE, tflite_graph_dump)
        ]
        # Get list of neutron subgraphs.
        self.neutron_subgraphs = self._get_neutron_subgraphs(neutron_graph_dump)
        if self.neutron_subgraphs:
            self._update_neutron_subgraphs_info(extracted_graph_dump)

    def _get_neutron_subgraphs(self, graph_dump: str) -> list[SubgraphInfo]:
        """Parse Neutron graph dump and extract subgraph information.

        :param graph_dump: String containing the Neutron graph dump from the converter log.
        :return: List of SubgraphInfo objects containing subgraph metadata and operator nodes.
        """

        def get_subgraph_nodes(subrgraph_dump: str) -> list[Node]:
            """Parse subgraph dump and extract operator nodes.

            :param subgraph_dump: String containing a single Neutron subgraph definition.
            :return: List of Node objects representing operators in the subgraph.
            """
            return [
                Node(
                    matched_operator.group("type"),
                    get_tensors_name(matched_operator.group("inputs")),
                    get_tensors_name(matched_operator.group("outputs")),
                    int(matched_operator.group("location")),
                )
                for matched_operator in re.finditer(PATTERN_NODE, subrgraph_dump)
            ]

        subgraphs = graph_dump.split(r"Name: subgraph_")
        if len(subgraphs) < 3:
            return []

        # Get numbers of final neutron graphs in converted model.
        self.neutron_graphs = [
            int(matched_graphs.group("num"))
            for matched_graphs in re.finditer(PATTERN_GRAPH, subgraphs[-1])
        ]
        if not self.neutron_graphs:
            return []

        # Get subgraphs
        neutron_subgraphs: list[SubgraphInfo] = []
        for subgraph in subgraphs[1:]:
            subgraph_match = re.search(PATTERN_SUBGRAPH, subgraph)
            if not subgraph_match:
                continue
            neutron_subgraph = SubgraphInfo(
                int(subgraph_match.group("num")),
                -1,
                get_tensors_name(subgraph_match.group("inputs")),
                get_tensors_name(subgraph_match.group("outputs")),
                0,
                get_subgraph_nodes(subgraph),
            )
            neutron_subgraphs.append(neutron_subgraph)
        return neutron_subgraphs

    def _update_neutron_subgraphs_info(self, extracted_graph: str) -> None:
        """Update Neutron subgraphs with verbose info.

        - Set numbers of Neutron kernels in each Neutron subgraph. 99% of subgraphs contain only one Neutron kernel,
        but there are some exceptions and some subgraphs can have more kernels. This number can be taken from
        final Neutron graph info.
        - Set Neutron subgraphs location in the final Neutron Graph. The function updates the location parameter
        for each Neutron subgraph according to its position in the final Neutron graph. Location is calculated
        continuously across all Neutron graphs in the model. Non-Neutron operators are skipped.

        :param extracted_graph: verbose Neutron graph dump.
        """
        # Neutron graphs.
        neutron_graphs = extracted_graph.split("NeutronGraph")
        location_shift = 0
        for neutron_graph in neutron_graphs:

            subgraph_nodes = {
                int(matched_subgraph.group("subgraph")): {
                    "location": i + location_shift,
                    "kernels": [
                        kernel.replace(" ", "")
                        for kernel in matched_subgraph.group("kernels").split("\n")
                        if kernel.strip()
                    ],
                }
                for i, matched_subgraph in enumerate(
                    re.finditer(PATTERN_VERBOSE_KERNELS, neutron_graph)
                )
            }
            if not subgraph_nodes:
                continue
            # Update location offset according to the number of kernels in the subgraph.
            location_shift += len(subgraph_nodes)

            # Neutron graphs.
            graph_num = -1
            matched_graph = re.search(r"subgraph_(?P<subgraph>\d+)", neutron_graph)
            if matched_graph:
                graph_num = int(matched_graph.group("subgraph"))

            # Update number of kernels for all subgraphs.
            for subgraph in self.neutron_subgraphs:
                if subgraph.num in subgraph_nodes:
                    subgraph.kernels = len(subgraph_nodes[subgraph.num]["kernels"])
                    subgraph.location = subgraph_nodes[subgraph.num]["location"]
                elif subgraph.num == graph_num:
                    subgraph.kernels = sum(
                        len(s["kernels"]) for s in subgraph_nodes.values()
                    )
                    self.neutron_kernels_num += subgraph.kernels

    def _nodes_match_by_io(self, tf_node: Node, neutron_node: Node) -> bool:
        """
        Determine whether a TFLite node can be mapped to a Neutron node
        based on their input and output compatibility.

        :param tf_node: Source TFLite node.
        :param neutron_node: Target Neutron node.
        :return: True if the nodes can be considered mapped, False otherwise.
        """

        def get_name_matches(tf_names: list[str], neutron_names: list[str]) -> int:
            # Count how many names from tf_names have a corresponding match in
            # neutron_names. A match is defined as:
            #   - exact equality, or
            #   - one name being a hierarchical variant of the other
            #     (i.e., sharing a common prefix separated by "/").
            result = 0
            for tf_name in tf_names:
                # Determine if the tensor name corresponds to a special operation input.
                # Matches names like "perm0", "perm1", etc. used by Transpose ops,
                # and names like "padding0", "padding1", etc. used by Pad ops.
                special_op = (
                    "permutation"
                    if re.fullmatch(r"perm(\d+)?", tf_name)
                    else (
                        "padding"
                        if re.fullmatch(r"padding(s)?(\d+)?", tf_name)
                        else None
                    )
                )
                for neutron_name in neutron_names:
                    if (
                        neutron_name == tf_name
                        or neutron_name + "/" in tf_name
                        or tf_name + "/" in neutron_name
                    ):
                        result += 1
                        break

                    # Check if the neutron input is also the special op (Pad or Transpose)
                    if special_op and special_op in neutron_name:
                        result += 1
                        break
            return result

        name_matches = get_name_matches(tf_node.inputs, neutron_node.inputs)
        # Map the node if all TFLite inputs match Neutron inputs.
        # Note: the Neutron node may still have additional extra inputs.
        if name_matches == len(tf_node.inputs):
            return True
        elif name_matches == len(tf_node.inputs) - 1:
            # If there is only one unmatched input, check matching of outputs.
            name_matches = get_name_matches(tf_node.outputs, neutron_node.outputs)
            if name_matches == len(tf_node.outputs):
                # Map the node if all TFLite outputs match Neutron outputs.
                return True
        return False

    def get_tflite_to_neutron_map(self) -> dict[int, tuple[int, ...]]:
        """Map TFLite nodes from the original model to Neutron nodes in the converted model.

        The mapping is built based on input and output tensor names. Neutron tensors may have
        exactly the same names or use the format "tflite_input/additional_name".

        :return: Dictionary mapping TFLite node indices to tuple of Neutron subgraph indices.
        """
        tflite_to_neutron_dict = {}
        for tf_idx, tf_node in enumerate(self.tflite_nodes):
            subgraph_idxs = []
            location_shift = 0
            for subgraph in self.neutron_subgraphs:
                if subgraph.num in self.neutron_graphs:
                    continue
                for neutron_node in subgraph.nodes:
                    if self._nodes_match_by_io(tf_node, neutron_node):
                        for kernel in range(subgraph.kernels):
                            subgraph_idxs.append(subgraph.location + location_shift + kernel)
                        break
                location_shift += max(subgraph.kernels - 1, 0)
            # Filter subgraph_idxs to avoid mapping multiple parallel single-input nodes that consume the
            # same input tensor into the same TFLite node.
            subgraph_idxs = self._filter_single_input_nodes(tf_node.name, subgraph_idxs)
            if subgraph_idxs:
                tflite_to_neutron_dict[tf_idx] = tuple(subgraph_idxs)

        self.tflite_to_neutron_map = tflite_to_neutron_dict
        return self.tflite_to_neutron_map

    def _filter_single_input_nodes(
        self, node_name: str, subgraph_loc: list[int]
    ) -> list[int]:
        """
        Filter the Neutron-to-TFLite mapping to avoid mapping multiple parallel single-input nodes
        that consume the same input tensor to a single TFLite node.

        The function checks whether the current TFLite node is a supported single-input node
        (as defined in SINGLE_INPUT_NODES) and whether it is mapped to multiple Neutron nodes.
        In such cases, it is possible that parallel single-input Neutron nodes were incorrectly
        mapped to the same TFLite node.

        If more than one single-input Neutron node is mapped, only one is kept in the mapping:
        the Neutron node whose operation name matches the operation name of the current TFLite node.

        :param node_name: Operation name of the current TFLite node.
        :param subgraph_loc: List of Neutron subgraph indices whose inputs correspond to the
                            input of the current TFLite node.
        :return: Filtered list of Neutron subgraph indices to be mapped to the current TFLite node.
        """
        # Check if there can be potential issue in mapping.
        if node_name in SINGLE_INPUT_NODES and len(subgraph_loc) > 1:
            single_in_nodes = []
            # Find all single-input nodes in subgraph_idxs.
            subgraphs = (
                subgraph
                for subgraph in self.neutron_subgraphs
                if subgraph.location in subgraph_loc
            )
            for subgraph in subgraphs:
                for neutron_node in subgraph.nodes:
                    if neutron_node.name in SINGLE_INPUT_NODES:
                        single_in_nodes.append((subgraph.location, neutron_node.name))
            if len(single_in_nodes) > 0:
                # Keep only the node with the matching name when multiple single-input nodes are present in subgraph_idxs.
                for subgraph_id, single_in_node_name in single_in_nodes:
                    if single_in_node_name == node_name:
                        return [subgraph_id]
                return []
        return subgraph_loc

    def get_edge_to_neutron_map(self) -> dict[int, tuple[int, ...]]:
        """Map Edge nodes to Neutron nodes.

        :return: Dictionary mapping Edge node handles to tuple of Neutron subgraph indices.
        """
        self.get_tflite_to_neutron_map()
        edge_to_neutron_dict = {}

        for edge_handle, tflite_indices in self.edge_to_tflite_map.items():
            neutron_nodes = set()
            for tf_node in tflite_indices:
                if tf_node in self.tflite_to_neutron_map:
                    neutron_nodes.update(self.tflite_to_neutron_map[tf_node])
            if neutron_nodes:
                edge_to_neutron_dict[edge_handle] = tuple(neutron_nodes)

        self.edge_to_neutron_map = edge_to_neutron_dict
        return self.edge_to_neutron_map

    def get_neutron_to_edge_map(self) -> dict[int, tuple[int, ...]]:
        """
        Transform edge-to-neutron map to neutron-to-edge map.

        :return: Dictionary mapping neutron_index to tuple of edge_handles
        """
        if not self.edge_to_neutron_map:
            _ = self.get_edge_to_neutron_map()

        neutron_to_edge = {}

        for edge_handle, neutron_indices in self.edge_to_neutron_map.items():
            for neutron_idx in neutron_indices:
                if neutron_idx not in neutron_to_edge:
                    neutron_to_edge[neutron_idx] = []
                neutron_to_edge[neutron_idx].append(edge_handle)

        # Fill gaps with empty tuples and convert lists to tuples.
        if neutron_to_edge:
            max_neutron_idx = self.neutron_kernels_num
            result = {}
            # Add one more non-mapped event at the end of list for the Neutron Dump event.
            for i in range(max_neutron_idx + 1):
                if i in neutron_to_edge:
                    result[i] = tuple(neutron_to_edge[i])
                else:
                    result[i] = ()
            logging.info(f"Neutron to Edge map was created: {result}")
            return result
        else:
            return {}
