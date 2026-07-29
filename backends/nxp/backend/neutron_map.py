# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import re
from collections import defaultdict
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
# Match a tensor name followed by its kind on the next non-empty line.
# example:
#   Name: quantized_decomposed_dequantize_per_tensor_default_5
#   Kind: Variable
PATTERN_TENSOR_KIND = r"Name:\s+(?P<name>\S+)\s+Kind:\s+(?P<kind>[^\r\n]+)"
# The pattern is very similar to operator pattern.
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
# Two graphs are expected in the input log: original and converted.
EXPECTED_GRAPHS = 2
# Marker for a Variable (dynamic) tensor kind in the converter log.
TENSOR_KIND_VARIABLE = "Variable"


@dataclass
class Node:
    name: str  # Name of the node/operator.
    inputs: list[str]  # Dynamic (variable) inputs only.
    outputs: list[str]  # Dynamic (variable) outputs only.
    location: int  # Location in graph/subgraph.


@dataclass
class SubgraphInfo:
    num: int  # Subgraph number.
    location: int  # Location in neutron graph.
    inputs: list[str]  # Dynamic inputs.
    outputs: list[str]  # Dynamic outputs.
    kernels: int  # Number of neutron kernels in this subgraph.
    nodes: list[Node]  # Operator nodes (for diagnostics).


def get_tensors_name(tensors: str) -> list[str]:
    """Parse a tensor-list string and return the tensor names."""
    return [m.group("name") for m in re.finditer(PATTERN_IO_TENSOR_NAME, tensors)]


def extract_tensor_kinds(graph_dump: str) -> dict[str, bool]:
    """Return a map of tensor name -> True (Variable/dynamic) or False (Constant/static).

    :param graph_dump: Raw text of a graph section from the Neutron converter log.
    """
    return {
        m.group("name"): m.group("kind").strip() == TENSOR_KIND_VARIABLE
        for m in re.finditer(PATTERN_TENSOR_KIND, graph_dump)
    }


def filter_dynamic(tensor_names: list[str], kinds: dict[str, bool]) -> list[str]:
    """Return only the dynamic (Variable) tensor names.

    Tensors absent from kinds are kept (no Kind annotation implies dynamic).

    :param tensor_names: Candidate tensor names.
    :param kinds: Map produced by extract_tensor_kinds().
    """
    return [n for n in tensor_names if kinds.get(n, True)]


def tensors_match(a: str, b: str) -> bool:
    """Return True if two tensor names refer to the same tensor.

    Matching strategy (in order):
      1. Exact equality.
      2. Hierarchical prefix: one name is a "/" - separated prefix of the other.
         Handles cases where the Neutron converter appends suffixes such as "/pad"
         or "/transpose" to the original name.

    Leaf-name matching is intentionally omitted - short generic names like "Relu"
    or "pad" appear in many unrelated tensors and cause false positives.

    :param a: First tensor name.
    :param b: Second tensor name.
    """
    if a == b:
        return True
    if b + "/" in a or a + "/" in b:
        return True
    return False


def count_tensor_matches(names_a: list[str], names_b: list[str]) -> int:
    """Count how many names in names_a have at least one match in names_b."""
    return sum(1 for a in names_a if any(tensors_match(a, b) for b in names_b))


class NeutronMap:
    """Mapping between Neutron, TFLite, and Edge operators based on the Neutron compiler log.

    Parses the Neutron converter log to extract TFLite nodes and Neutron subgraphs, then
    maps TFLite operators to Neutron operators using dynamic (variable) I/O tensor names.

    Matching operates at the Neutron subgraph chain level, which is robust to internal
    optimizer transformations and supports all mapping cardinalities (1-to-1, many-to-1,
    1-to-many, many-to-many).

    Attributes:
        tflite_nodes (list[Node]): TFLite node information (dynamic I/O only).
        neutron_subgraphs (list[SubgraphInfo]): Neutron subgraph information.
        neutron_graphs (list[int]): Numbers of top-level Neutron graphs.
        neutron_kernels_num (int): Total number of Neutron kernels.
        edge_to_tflite_map (dict[int, tuple[int, ...]]): Edge -> TFLite operator map.
        tflite_to_neutron_map (dict[int, tuple[int, ...]]): TFLite -> Neutron operator map.
        edge_to_neutron_map (dict[int, tuple[int, ...]]): Edge -> Neutron operator map.

    Example:
        >>> nmap = NeutronMap(log_output, edge_to_tflite_map)
        >>> neutron_to_edge = nmap.get_neutron_to_edge_map()
    """

    def __init__(
        self, neutron_compiler_log: str, edge_to_tflite_map: dict[int, tuple[int, ...]]
    ) -> None:
        """Initialize neutron map from neutron compiler log.

        :param neutron_compiler_log: Log text with dumpAfterImport and dumpAfterGenerate
            set to "console" so TFLite and Neutron graph dumps are present.
        :param edge_to_tflite_map: Edge operator index -> tuple of TFLite operator indices.
        """
        self.tflite_nodes: list[Node] = []
        self.neutron_subgraphs: list[SubgraphInfo] = []
        self.neutron_graphs: list[int] = []
        self.neutron_kernels_num: int = 0
        self.edge_to_tflite_map = edge_to_tflite_map
        self.tflite_to_neutron_map: dict[int, tuple[int, ...]] = {}
        self.edge_to_neutron_map: dict[int, tuple[int, ...]] = {}
        self._tflite_tensor_names: set[str] = set()
        self._split_profiling_log(neutron_compiler_log)

    # ------------------------------------------------------------------
    # Log parsing
    # ------------------------------------------------------------------

    def _split_profiling_log(self, log: str) -> None:
        """Parse the compiler log and populate tflite_nodes and neutron_subgraphs."""
        graphs = log.split("Graphs:")
        if len(graphs) != EXPECTED_GRAPHS + 1:
            return
        optimization_dump, neutron_graph_dump = graphs[1:]

        tflite_graph_dump = optimization_dump.partition("= Optimize Graph =")[0]
        extracted_graph_dump = optimization_dump.partition("= Extract Graphs =")[
            2
        ].partition("Generate code for NeutronGraph")[0]

        tflite_kinds = extract_tensor_kinds(tflite_graph_dump)
        self.tflite_nodes = [
            Node(
                m.group("type"),
                filter_dynamic(get_tensors_name(m.group("inputs")), tflite_kinds),
                filter_dynamic(get_tensors_name(m.group("outputs")), tflite_kinds),
                int(m.group("location")),
            )
            for m in re.finditer(PATTERN_NODE, tflite_graph_dump)
        ]

        # Collect TFLite tensor names (dynamic I/O of every TFLite node).
        # Used in _is_neutron_consumer to distinguish TFLite-boundary tensors from
        # Neutron-internal tensors and prevent false-positive hierarchical chaining.
        self._tflite_tensor_names = {
            t for n in self.tflite_nodes for t in n.inputs + n.outputs
        }

        # Cache lookup structures for _find_matching_tflite_chain so they are
        # built once per NeutronMap instance rather than on every chain search.
        self._node_by_loc: dict[int, Node] = {n.location: n for n in self.tflite_nodes}
        self._input_to_locs: dict[str, list[int]] = defaultdict(list)
        for _n in self.tflite_nodes:
            for _inp in _n.inputs:
                self._input_to_locs[_inp].append(_n.location)

        neutron_kinds = extract_tensor_kinds(neutron_graph_dump)
        self.neutron_subgraphs = self._parse_neutron_subgraphs(
            neutron_graph_dump, neutron_kinds
        )
        if self.neutron_subgraphs:
            self._update_neutron_subgraphs_info(extracted_graph_dump)

    def _parse_neutron_subgraphs(
        self, graph_dump: str, tensor_kinds: dict[str, bool]
    ) -> list[SubgraphInfo]:
        """Parse the Neutron graph dump and return subgraph metadata with dynamic I/O only.

        :param graph_dump: String containing the Neutron graph dump from the compiler log.
        :return: List of SubgraphInfo objects containing subgraph metadata and operator nodes.
        """

        def parse_nodes(subgraph_dump: str) -> list[Node]:
            return [
                Node(
                    m.group("type"),
                    filter_dynamic(get_tensors_name(m.group("inputs")), tensor_kinds),
                    filter_dynamic(get_tensors_name(m.group("outputs")), tensor_kinds),
                    int(m.group("location")),
                )
                for m in re.finditer(PATTERN_NODE, subgraph_dump)
            ]

        sections = graph_dump.split(r"Name: subgraph_")
        if len(sections) < 3:
            return []

        self.neutron_graphs = [
            int(m.group("num")) for m in re.finditer(PATTERN_GRAPH, sections[-1])
        ]
        if not self.neutron_graphs:
            return []

        subgraphs: list[SubgraphInfo] = []
        for section in sections[1:]:
            m = re.search(PATTERN_SUBGRAPH, section)
            if not m:
                continue
            subgraphs.append(
                SubgraphInfo(
                    num=int(m.group("num")),
                    location=-1,
                    inputs=filter_dynamic(
                        get_tensors_name(m.group("inputs")), tensor_kinds
                    ),
                    outputs=filter_dynamic(
                        get_tensors_name(m.group("outputs")), tensor_kinds
                    ),
                    kernels=0,
                    nodes=parse_nodes(section),
            )
            )
        return subgraphs

    def _update_neutron_subgraphs_info(self, extracted_graph: str) -> None:
        """Fill in location and kernel count for each Neutron subgraph from verbose output.

        :param extracted_graph: Verbose Neutron graph dump (Extract Graphs section).
        """
        location_shift = 0
        for graph_text in extracted_graph.split("NeutronGraph"):
            node_info: dict[int, dict] = {}
            running_loc = location_shift
            for m in re.finditer(PATTERN_VERBOSE_KERNELS, graph_text):
                kernels = [k for k in m.group("kernels").split("\n") if k.strip()]
                node_info[int(m.group("subgraph"))] = {
                    "location": running_loc,
                    "kernels": kernels,
                }
                running_loc += len(kernels)
            if not node_info:
                continue
            location_shift = running_loc

            graph_num = -1
            gm = re.search(r"subgraph_(?P<subgraph>\d+)", graph_text)
            if gm:
                graph_num = int(gm.group("subgraph"))

            for sg in self.neutron_subgraphs:
                if sg.num in node_info:
                    sg.kernels = len(node_info[sg.num]["kernels"])
                    sg.location = node_info[sg.num]["location"]
                elif sg.num == graph_num:
                    sg.kernels = sum(len(v["kernels"]) for v in node_info.values())
                    self.neutron_kernels_num += sg.kernels

    # ------------------------------------------------------------------
    # Neutron subgraph chain helpers
    # ------------------------------------------------------------------

    def _is_neutron_consumer(
        self, consumer: SubgraphInfo, producer: SubgraphInfo
    ) -> bool:
        """Return True if consumer directly follows producer in a Neutron subgraph chain.

        A connecting tensor must satisfy two conditions:
          1. tensors_match() confirms the producer output and consumer input are related.
          2. BOTH the producer output and consumer input must be absent from the original
             TFLite tensor name set (i.e. they are Neutron-internal tensors).
             This dual check is required because tensors_match() uses hierarchical
             containment. For example, "CifarNet/logits/BiasAdd" (TFLite-level output)
             would hierarchically match "CifarNet/logits/BiasAdd/pad" (Neutron-internal
             input), which would wrongly chain subgraphs across TFLite boundaries.
        """
        if not consumer.inputs or not producer.outputs:
            return False
        connecting_pairs = [
            (out, inp)
            for out in producer.outputs
            for inp in consumer.inputs
            if tensors_match(inp, out)
        ]
        if not connecting_pairs:
            return False
        # Pass-through subgraph (input name == output name): always chain after predecessor.
        # These are injected by the Neutron converter purely for data routing.
        if consumer.inputs == consumer.outputs:
            return True
        # Both sides of each connecting pair must be Neutron-internal (not TFLite tensors).
        return all(
            out not in self._tflite_tensor_names
            and inp not in self._tflite_tensor_names
            for out, inp in connecting_pairs
                    )

    def _get_neutron_subgraph_chains(self) -> list[list[SubgraphInfo]]:
        """Group active Neutron subgraphs into linearly-connected execution chains.

        A chain is a sequence [sg_0, sg_1, ...] where each sg_{i+1} is the unique
        consumer of sg_i's outputs. Subgraphs not part of a longer chain form singletons.
        """
        active = sorted(
            (
                sg
                for sg in self.neutron_subgraphs
                if sg.num not in self.neutron_graphs and sg.location >= 0
            ),
            key=lambda sg: sg.location,
        )

        has_predecessor: set[int] = {
            sg.num
            for sg in active
            for other in active
            if other is not sg and self._is_neutron_consumer(sg, other)
        }

        chains: list[list[SubgraphInfo]] = []
        visited: set[int] = set()

        for start in active:
            if start.num in visited or start.num in has_predecessor:
                continue
            chain = [start]
            visited.add(start.num)
            while True:
                successors = [
                    sg
                    for sg in active
                    if sg.num not in visited
                    and self._is_neutron_consumer(sg, chain[-1])
                ]
                if len(successors) != 1:
                    break
                chain.append(successors[0])
                visited.add(successors[0].num)
            chains.append(chain)

        # Any subgraphs unreachable from a chain root become singletons.
        for sg in active:
            if sg.num not in visited:
                chains.append([sg])

        return chains

    def _get_chain_boundary_io(
        self, chain: list[SubgraphInfo]
    ) -> tuple[list[str], list[str]]:
        """Return the external (boundary) inputs and outputs of a Neutron subgraph chain.

        Chain inputs = inputs of the first subgraph.
        Chain outputs = outputs not consumed internally by any later subgraph.
        """
        if not chain:
            return [], []

        chain_inputs = list(chain[0].inputs)
        chain_outputs = list(chain[0].outputs)

        for sg in chain[1:]:
            consumed = set(sg.inputs)
            chain_outputs = [
                o
                for o in chain_outputs
                if not any(tensors_match(o, c) for c in consumed)
            ]
            chain_outputs.extend(sg.outputs)

        return chain_inputs, chain_outputs

    # ------------------------------------------------------------------
    # TFLite chain matching helpers
    # ------------------------------------------------------------------

    def _inputs_match(self, sg_inputs: list[str], tf_node: Node) -> bool:
        """Return True if all dynamic inputs of tf_node are covered by sg_inputs."""
        return (
            bool(tf_node.inputs)
            and bool(sg_inputs)
            and (count_tensor_matches(tf_node.inputs, sg_inputs) == len(tf_node.inputs))
                    )

    def _outputs_match(self, sg_outputs: list[str], chain_outputs: list[str]) -> bool:
        """Return True if all chain_outputs are covered by sg_outputs."""
        return (
            bool(chain_outputs)
            and bool(sg_outputs)
            and (count_tensor_matches(chain_outputs, sg_outputs) == len(chain_outputs))
                )

    def _find_matching_tflite_chain(
        self, sg_inputs: list[str], sg_outputs: list[str]
    ) -> list[int]:
        """Find the TFLite operator chain whose collective I/O matches the given boundary I/O.

        Algorithm:
          1. Identify candidate starting nodes whose dynamic inputs match sg_inputs.
          2. For each candidate, check for a 1-to-1 output match.
          3. Otherwise, extend the chain forward greedily until the outputs match sg_outputs.

        :param sg_inputs: Dynamic inputs of the (virtual) Neutron subgraph/chain.
        :param sg_outputs: Dynamic outputs of the (virtual) Neutron subgraph/chain.
        :return: Ordered list of TFLite node locations forming the match, or [].
        """
        for start in (n for n in self.tflite_nodes if self._inputs_match(sg_inputs, n)):
            chain_locs = [start.location]
            chain_outs = list(start.outputs)

            if self._outputs_match(sg_outputs, chain_outs):
                return chain_locs

            for _ in range(len(self.tflite_nodes)):
                next_loc = self._find_next_chain_node(
                    chain_locs, chain_outs, self._node_by_loc, self._input_to_locs
                            )
                if next_loc is None:
                        break
                next_node = self._node_by_loc[next_loc]
                chain_locs.append(next_loc)
                consumed = set(next_node.inputs)
                chain_outs = [
                    o for o in chain_outs if o not in consumed
                ] + next_node.outputs
                if self._outputs_match(sg_outputs, chain_outs):
                    return chain_locs

        return []

    def _find_next_chain_node(
        self,
        chain_locs: list[int],
        chain_outputs: list[str],
        node_by_loc: dict[int, Node],
        input_to_locs: dict[str, list[int]],
    ) -> int | None:
        """Return the unique next TFLite node that can extend the current chain, or None.

        A node is eligible only if all its dynamic inputs are covered by the current chain
        outputs (no external dependency). A fork (multiple eligible nodes) returns None.
        """
        chain_loc_set = set(chain_locs)
        chain_out_set = set(chain_outputs)
        eligible: set[int] = set()

        for out_name in chain_outputs:
            # Exact-name consumers first.
            consumers = [
                loc
                for loc in input_to_locs.get(out_name, [])
                if loc not in chain_loc_set
            ]
            # Fallback: hierarchical match (handles Neutron-renamed tensors).
            if not consumers:
                consumers = [
                    n.location
                    for n in self.tflite_nodes
                    if n.location not in chain_loc_set
                    and any(tensors_match(out_name, inp) for inp in n.inputs)
                ]
            for loc in consumers:
                candidate = node_by_loc[loc]
                if all(
                    any(tensors_match(inp, co) for co in chain_out_set)
                    for inp in candidate.inputs
                ):
                    eligible.add(loc)

        return next(iter(eligible)) if len(eligible) == 1 else None

    # ------------------------------------------------------------------
    # Public mapping API
    # ------------------------------------------------------------------

    def get_tflite_to_neutron_map(self) -> dict[int, tuple[int, ...]]:
        """Map TFLite node locations to Neutron kernel indices.

        Neutron subgraphs are first grouped into chains; each chain is matched as a unit
        against a TFLite operator chain using dynamic I/O tensor names.

        :return: Dict: TFLite node location -> tuple of Neutron kernel indices.
        """
        result: dict[int, set[int]] = {}

        for chain in self._get_neutron_subgraph_chains():
            chain_inputs, chain_outputs = self._get_chain_boundary_io(chain)
            if not chain_inputs or not chain_outputs:
                continue

            neutron_indices = [
                idx
                for sg in chain
                for idx in range(sg.location, sg.location + max(sg.kernels, 1))
            ]

            tflite_locs = self._find_matching_tflite_chain(chain_inputs, chain_outputs)
            if not tflite_locs:
                logging.debug(
                    f"No TFLite match for Neutron chain {[sg.num for sg in chain]} "
                    f"(inputs={chain_inputs}, outputs={chain_outputs})"
            )
                continue

            for loc in tflite_locs:
                result.setdefault(loc, set()).update(neutron_indices)

        self.tflite_to_neutron_map = {k: tuple(sorted(v)) for k, v in result.items()}
        return self.tflite_to_neutron_map

    def get_edge_to_neutron_map(self) -> dict[int, tuple[int, ...]]:
        """Map Edge node handles to Neutron kernel indices.

        :return: Dict: Edge handle -> tuple of Neutron kernel indices.
        """
        self.get_tflite_to_neutron_map()
        result: dict[int, tuple[int, ...]] = {}
        for edge_handle, tflite_indices in self.edge_to_tflite_map.items():
            neutron = {
                n
                for tf_idx in tflite_indices
                for n in self.tflite_to_neutron_map.get(tf_idx, ())
            }
            if neutron:
                result[edge_handle] = tuple(neutron)
        self.edge_to_neutron_map = result
        return result

    def get_neutron_to_edge_map(self) -> dict[int, tuple[int, ...]]:
        """Return the inverse of the Edge-to-Neutron map.

        :return: Dict: Neutron kernel index -> tuple of Edge handles.
                 All indices up to neutron_kernels_num are present (empty tuple if unmapped).
                 One extra entry is added for the Neutron Dump event at the end.
        """
        if not self.edge_to_neutron_map:
            self.get_edge_to_neutron_map()

        inverse: dict[int, list[int]] = defaultdict(list)
        for edge_handle, neutron_indices in self.edge_to_neutron_map.items():
            for idx in neutron_indices:
                inverse[idx].append(edge_handle)

        if not inverse:
            return {}

        result = {
            i: tuple(inverse.get(i, ())) for i in range(self.neutron_kernels_num + 1)
        }
            logging.info(f"Neutron to Edge map was created: {result}")
            return result
