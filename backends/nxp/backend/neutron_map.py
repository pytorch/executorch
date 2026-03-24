#
# Copyright 2026 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
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
PATTERN_NODE = (r"Type:\s+(?P<type>\w+)\s+"
                r"Inputs:(?P<inputs>[\s\S]*?)"
                r"Outputs:(?P<outputs>[\s\S]*?)"
                r"Location:\s+(?P<location>\d+)")
# The pattern is very similar to operator pattern
PATTERN_SUBGRAPH = (r"^(?P<num>\d+)\s*"
                    r"Inputs:(?P<inputs>[\s\S]*?)"
                    r"Outputs:(?P<outputs>[\s\S]*?)"
                    r"Tensors:")
# example:  [0]: quantized_decomposed_quantize_per_tensor_default_4
PATTERN_IO_TENSOR_NAME = r"\[\d+\]:\s+(?P<name>[\S]+)"
# example: Statistics for NeutronGraph "subgraph_195":
PATTERN_GRAPH = r"Statistics for NeutronGraph \"subgraph_(?P<num>\d+)\":"
# example:  NeutronOperator "subgraph_011":
#                Operators:
#                    PADV2
#                Kernels:
#                    Pad
PATTERN_VERBOSE_KERNELS = (r"\"subgraph_(?P<subgraph>\d+)\"\:\s*"
                           r"Operators:[\s\S]*?"
                           r"Kernels:\s*(?P<kernels>[\s\S]*?)"
                           r"\s*(NeutronOperator|^$|=)")
# example:  NeutronGraph "subgraph_074":
PATTERN_VERBOSE_GRAPH = r"NeutronGraph\s*\"subgraph_(?P<subgraph>\d+)\":(?P<operators>[\s\S]*?)\s*(^$|=)"
# Two graphs are expected in the input log: original and converted.
EXPECTED_GRAPHS = 2

@dataclass
class Node:
    name: str # Name of the node.
    inputs: list[str] # List of nodes inputs.
    outputs: list[str] # List of nodes outputs.
    location: int # Location in graph/subgraph.

@dataclass
class SubgraphInfo:
    num: int # Subgraph number.
    location: int # Location in neutron graph
    inputs: list[str] # List of subgraphs inputs.
    outputs: list[str] # List of subgraphs outputs.
    kernels: int # Number of neutron kernels in neutron subgraph.
    nodes: list[Node] # List of tflite nodes in neutron subgraph.

def get_tensors_name(tensors: str) -> list[str]:
    """Split input string with tensor names into list of names"""
    return [m.group("name") for m in re.finditer(PATTERN_IO_TENSOR_NAME, tensors)]

class NeutronMap:
    """

    [name, inputs[], outputs[]]

    """
    pte_nodes: list[Node]
    tflite_nodes: list[Node]
    neutron_subgraphs: list[SubgraphInfo]
    neutron_graphs: list[int]
    edge_to_tflite_map: dict[int, tuple[int, ...]]
    edge_to_neutron_map: dict[int, tuple[int, ...]]
    tflite_to_neutron_map: dict[int, tuple[int, ...]]

    def __init__(self, neutron_converter_log: str, edge_to_tflite_map: dict[int, tuple[int, ...]]) -> None:
        """Initialize neutron map from neutron converter log.

        :param neutron_converter_log: neutron converter log obtained during model conversion. It should contain
        original tflite graph and neutron graph dump. To add these dumps to converter log the dumpAfterImport and
        dumpAfterGenerate flags have to be set to "console".
        """
        self.pte_nodes = []
        self.tflite_nodes = []
        self.neutron_subgraphs = []
        self.neutron_graphs = []
        self.edge_to_tflite_map = edge_to_tflite_map
        self.tflite_to_neutron_map = {}
        self.edge_to_neutron_map = {}
        super().__init__()
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
        extracted_graph_dump = (
            optimization_dump
            .partition("= Extract Graphs =")[2]
            .partition("Generate code for NeutronGraph")[0])

        # Get list of original operators from first dumped graph.
        self.tflite_nodes = [Node(
            matched_operator.group("type"),
            get_tensors_name(matched_operator.group("inputs")),
            get_tensors_name(matched_operator.group("outputs")),
            int(matched_operator.group("location"))
        )
            for matched_operator in re.finditer(PATTERN_NODE, tflite_graph_dump)
        ]
        # Get list of neutron subgraphs.
        self.neutron_subgraphs = self._get_neutron_subgraphs(neutron_graph_dump)
        if self.neutron_subgraphs:
            self._update_kernels_number(extracted_graph_dump)
            self._sort_neutron_subgraphs()


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
            return [Node(
                matched_operator.group("type"),
                get_tensors_name(matched_operator.group("inputs")),
                get_tensors_name(matched_operator.group("outputs")),
                int(matched_operator.group("location"))
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
                1,
                get_subgraph_nodes(subgraph)
            )
            neutron_subgraphs.append(neutron_subgraph)
        return neutron_subgraphs


    def _sort_neutron_subgraphs(self) -> None:
        """Sort Neutron subgraphs according to Neutron graph definitions.

        The function updates the location parameter for each Neutron subgraph according to its position
        in the final Neutron graph. Location is calculated continuously across all Neutron graphs in
        the model. Non-Neutron operators are skipped.
        """
        # Get order of subgraphs in neutron graph.
        location_shift = 0
        for num in self.neutron_graphs:
            neutron_graph = next((subgraph for subgraph in self.neutron_subgraphs if subgraph.num == num), None)
            if not neutron_graph:
                continue
            for node in neutron_graph.nodes:
                if node.name != "NeutronOperator":
                    # Skip non-Neutron operators when counting. Adjust location offset.
                    location_shift -= 1
                    continue
                for idx, subgraph in enumerate(self.neutron_subgraphs):
                    if subgraph.inputs == node.inputs and subgraph.outputs == node.outputs:
                        # Update location offset according to the number of kernels in the subgraph.
                        # Mup only the last kernel in subgraph.
                        location_shift += self.neutron_subgraphs[idx].kernels - 1
                        # Update subgraph location with the real position in neutron graph.
                        self.neutron_subgraphs[idx].location = node.location + location_shift
                        break
            # Adjust location offset if there is more than one Neutron graph.
            location_shift += len(neutron_graph.nodes)


    def _update_kernels_number(self, extracted_graph: str) -> None:
        """Set number of Neutron kernels for each Neutron subgraphs according to the verbose Neutron graph dump.

        99% of subgraphs contain only one Neutron kernel, but there are some exceptions and some subgraphs can
        have more kernels.
        :param extracted_graph: verbose Neutron graph dump.
        :return: sets the kernels parameter for each subgraphs from self.neutron_subgraphs.
        """
        # Neutron subgraphs
        subgraph_nodes = {
            int(matched_subgraph.group("subgraph")): [
                kernel.replace(" ", "")
                for kernel in matched_subgraph.group("kernels").split("\n")
                if kernel.strip()
            ]
            for matched_subgraph in re.finditer(PATTERN_VERBOSE_KERNELS, extracted_graph)
        }
        # Neutron graphs
        graph_nodes = {
            int(matched_graph.group("subgraph")):
                matched_graph.group("operators").count("NeutronOperator")
            for matched_graph in re.finditer(PATTERN_VERBOSE_GRAPH, extracted_graph)
        }
        # Update nuber of kernels for all subgraphs
        for subgraph in self.neutron_subgraphs:
            if subgraph.num in subgraph_nodes:
                subgraph.kernels = len(subgraph_nodes[subgraph.num])
            elif subgraph.num in graph_nodes:
                subgraph.kernels = graph_nodes[subgraph.num]


    def get_tflite_to_neutron_map(self) -> dict[int, tuple[int, ...]]:
        """Map TFLite nodes from the original model to Neutron nodes in the converted model.

        :return: Dictionary mapping TFLite node indices to tuple of Neutron subgraph indices.
        """
        tflite_to_neutron_dict = {}
        for tf_idx, tf_node in enumerate(self.tflite_nodes):
            subgraph_idxs = []
            for subgraph in self.neutron_subgraphs:
                if subgraph.num in self.neutron_graphs or subgraph.location in subgraph_idxs:
                    continue
                for neutron_node in subgraph.nodes:
                    name_matches = 0
                    for tf_in in tf_node.inputs:
                        for neutron_in in neutron_node.inputs:
                            if neutron_in == tf_in or neutron_in + "/" in tf_in or tf_in + "/" in neutron_in:
                                name_matches += 1
                                break
                    if name_matches == len(tf_node.inputs):
                        # Map neutron node to tflite node if inputs of tflite node maps to neutron node
                        subgraph_idxs.append(subgraph.location)
                    elif name_matches == len(tf_node.inputs) - 1:
                        # Else check outputs
                        name_matches = 0
                        for tf_out in tf_node.outputs:
                            for neutron_out in neutron_node.outputs:
                                if neutron_out == tf_out or neutron_out + "/" in tf_out or tf_out + "/" in neutron_out:
                                    name_matches += 1
                                    break
                        if name_matches == len(tf_node.outputs):
                            subgraph_idxs.append(subgraph.location)
            if subgraph_idxs:
                tflite_to_neutron_dict[tf_idx] = tuple(subgraph_idxs)

        self.tflite_to_neutron_map = tflite_to_neutron_dict
        return self.tflite_to_neutron_map

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

        # Fill gaps with empty tuples and convert lists to tuples
        if neutron_to_edge:
            max_neutron_idx = max(neutron_to_edge.keys())
            result = {}
            # Add one more non-mapped event at the end of list for the Neutron Dump event
            for i in range(max_neutron_idx + 2):
                if i in neutron_to_edge:
                    result[i] = tuple(neutron_to_edge[i])
                else:
                    result[i] = ()
            logging.info(f"Neutron to Edge map was created: {result}")
            return result
        else:
            return {}