# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

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
# The pattern is very similar to the operator pattern.
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
# example:   numKernelCalls      0x9
PATTERN_NUM_KERNEL_CALLS = r"numKernelCalls\s+0x([0-9a-fA-F]+)"
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
# Regex to extract stable weight pointer offsets from a CALLARGS parameter string.
# Matches filterPtr, biasPtr, outPostScalePtr fields — offsets into the Consts region
# that remain identical across all batch repetitions of the same operator but differ
# between sequential operators that happen to share the same kernel type.
_WEIGHT_PTR_RE = re.compile(r"(?:filterPtr|biasPtr|outPostScalePtr):\s*\(([^)]+)\)")

TENSOR_KIND_VARIABLE = "Variable"
# Two graphs are expected in the input log: original (TFLite) and converted (Neutron).
EXPECTED_GRAPHS = 2


@dataclass
class Node:
    name: str
    inputs: list[str]  # dynamic (Variable) inputs only
    outputs: list[str]  # dynamic (Variable) outputs only
    location: int  # operator location index in its subgraph


@dataclass
class SubgraphInfo:
    num: int  # subgraph number as it appears in the converter log
    location: int  # CALLARGS slot index of the first kernel (batch-aware)
    inputs: list[str]  # dynamic (Variable) inputs
    outputs: list[str]  # dynamic (Variable) outputs
    kernels: int  # number of CALLARGS slots assigned to this subgraph
    nodes: list[Node]  # operator nodes inside this subgraph (for diagnostics)
    kernel_names: list[str] = field(
        default_factory=list
    )  # kernel names from Extract Graphs


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

    Leaf-name matching is intentionally omitted because short generic names like
    "Relu" or "pad" appear in many unrelated tensors and cause false positives.

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
        neutron_kernels_num (int): Total number of Neutron kernels (runtime CALLARGS count).
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
        # Set to True when neutron_kernels_num was overridden from the microcode
        # numKernelCalls field (authoritative for batch > 1 / tiling). In that case
        # the completeness sanity check is enabled.
        self._microcode_kernels_authoritative: bool = False
        # Number of CALLARGS slots assigned to NeutronOperators by
        # _remap_locations_from_microcode. Excludes injected helper slots (MemCpy, etc.)
        # that have no TFLite counterpart and can never be covered by the mapping.
        self._coverable_slot_count: int = 0
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
            # Remap subgraph locations and kernel counts from the actual CALLARGS sequence.
            # This fixes batch > 1 where each operator generates N CALLARGS events and
            # handles injected helper kernels (MemCpy, extra StridedSliceConcat) that are
            # not listed in Extract Graphs but do appear in the Microinstructions section.
            self._remap_locations_from_microcode(optimization_dump)

        # Override neutron_kernels_num with the value from the microcode header when
        # available. The microcode numKernelCalls field counts the actual CALLARGS
        # instructions executed at runtime. When it exceeds the Extract Graphs count
        # the microcode repeats or injects kernels (batch > 1, tiling, helper ops)
        # and the completeness sanity check becomes meaningful.
        microcode_kernel_calls = self._parse_num_kernel_calls(optimization_dump)
        if microcode_kernel_calls > self.neutron_kernels_num:
            self.neutron_kernels_num = microcode_kernel_calls
            self._microcode_kernels_authoritative = True

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

        Parses the Extract Graphs section (verbose kernel listing) to determine each
        NeutronOperator's batch=1 kernel offset and kernel name list. The top-level
        NeutronGraph entry accumulates the total kernel count into neutron_kernels_num.

        :param extracted_graph: Verbose Neutron graph dump (Extract Graphs section).
        """
        location_shift = 0
        for graph_text in extracted_graph.split("NeutronGraph"):
            node_info: dict[int, dict] = {}
            running_loc = location_shift
            for m in re.finditer(PATTERN_VERBOSE_KERNELS, graph_text):
                # strip() is required: the Extract Graphs section uses Windows-style
                # line endings (\r\n) and indented kernel names, so raw splits yield
                # entries like "Pad\r" or "            Conv2DStandardV2".
                kernels = [
                    k.strip() for k in m.group("kernels").split("\n") if k.strip()
                ]
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
                    sg.kernel_names = node_info[sg.num]["kernels"]
                elif sg.num == graph_num:
                    # Top-level NeutronGraph entry: accumulate total kernel count.
                    sg.kernels = sum(len(v["kernels"]) for v in node_info.values())
                    self.neutron_kernels_num += sg.kernels

    @staticmethod
    def _parse_num_kernel_calls(optimization_dump: str) -> int:
        """Return the sum of numKernelCalls across all NeutronGraphs in the microcode header.

        :param optimization_dump: Converter log section between 'Graphs:' splits.
        :return: Total CALLARGS instruction count, or 0 if the field is absent (older logs).
        """
        return sum(
            int(m.group(1), 16)
            for m in re.finditer(PATTERN_NUM_KERNEL_CALLS, optimization_dump)
        )

    @staticmethod
    def _parse_callargs(
        optimization_dump: str,
    ) -> list[tuple[str, tuple[str, ...] | None]]:
        """Parse CALLARGS entries from the Microinstructions section.

        Returns (kernel_name, fingerprint) pairs where fingerprint is a tuple of
        weight-pointer offsets (filterPtr, biasPtr, outPostScalePtr) used to distinguish
        same-type operators across batch repetitions. None when no weight pointers are present.

        :param optimization_dump: Converter log section between 'Graphs:' splits.
        :return: List of (kernel_name, fingerprint) pairs in CALLARGS order.
        """
        result = [
            (
                m.group(1),
                (lambda h: tuple(h) if h else None)(_WEIGHT_PTR_RE.findall(m.group(2))),
            )
            for m in re.finditer(
                r"\bCALLARGS\s+(\w+)\s+@\(\)\s+\{([^}]*)\}", optimization_dump
            )
        ]
        if not result:
            result = [
                (m.group(1), None)
                for m in re.finditer(r"\bCALLARGS\s+(\w+)\b", optimization_dump)
            ]
        return result

    @staticmethod
    def _assign_group_slots(
        group: list[SubgraphInfo],
        group_entries: list[tuple[int, tuple[str, ...] | None]],
    ) -> None:
        """Assign CALLARGS slot indices to a group of same-first-kernel subgraphs.

        Uses weight-pointer fingerprints to distinguish two cases:
          - Each subgraph has a unique fingerprint: sequential operators of the same
            kernel type. Each fingerprint bucket is assigned to one subgraph.
          - All entries share one fingerprint (or fingerprint count != group size):
            batch repetitions of the same operator, or an ambiguous case. Slots are
            divided evenly among group members.

        After assignment, sg.location holds the absolute CALLARGS slot index of the
        first kernel call for that subgraph, and sg.kernels holds the total count.

        :param group: Subgraphs to assign slots to (a contiguous slice of active[]).
        :param group_entries: (absolute_slot_index, fingerprint) pairs for this group.
        """
        fp_buckets: dict[tuple[str, ...] | None, list[int]] = {}
        for abs_idx, fp in group_entries:
            fp_buckets.setdefault(fp, []).append(abs_idx)

        group_size = len(group)
        if len(fp_buckets) == group_size and group_size > 1:
            # Each subgraph has a unique fingerprint: assign its own bucket.
            for sg, indices in zip(group, fp_buckets.values()):
                sg.location = indices[0]
                # Each CALLARGS hit counts only the first kernel. The total slot count for
                # this subgraph is repetitions * kernels_per_subgraph. For batch=1 each
                # bucket has exactly 1 entry so kernels stays as set by Extract Graphs.
                sg.kernels = len(indices) * max(len(sg.kernel_names), sg.kernels)
        else:
            # Shared fingerprint (batch repetitions) or count mismatch: divide evenly.
            total = len(group_entries)
            per = max(total // group_size, 1)
            slices = [
                group_entries[m * per : (m + 1) * per] for m in range(group_size - 1)
            ]
            slices.append(group_entries[(group_size - 1) * per :])
            for sg, sl in zip(group, slices):
                if sl:
                    sg.location = sl[0][0]
                    sg.kernels = len(sl) * max(len(sg.kernel_names), sg.kernels)

    def _remap_locations_from_microcode(self, optimization_dump: str) -> None:
        """Remap each active subgraph's location and kernel count using the actual CALLARGS sequence.

        No-op when no CALLARGS are found (older log format without Microinstructions section).

        :param optimization_dump: Converter log section between 'Graphs:' splits.
        """
        callargs_full = self._parse_callargs(optimization_dump)
        if not callargs_full:
            return

        callargs = [k for k, _ in callargs_full]

        # Active operator subgraphs with kernel name info, sorted by batch=1 location.
        active = sorted(
            [
                sg
                for sg in self.neutron_subgraphs
                if sg.num not in self.neutron_graphs
                and sg.location >= 0
                and sg.kernel_names
            ],
            key=lambda sg: sg.location,
        )
        if not active:
            return

        i = 0
        ca_idx = 0
        while i < len(active):
            first_kernel = active[i].kernel_names[0]

            # Find the end of the group of subgraphs sharing the same first kernel name.
            group_end = i + 1
            while (
                group_end < len(active)
                and active[group_end].kernel_names
                and active[group_end].kernel_names[0] == first_kernel
            ):
                group_end += 1

            # Advance to the first CALLARGS slot matching this group's first kernel.
            while ca_idx < len(callargs) and callargs[ca_idx] != first_kernel:
                ca_idx += 1
            if ca_idx >= len(callargs):
                break

            # Collect all entries for this group, stopping at the next group's first kernel.
            next_first = (
                active[group_end].kernel_names[0]
                if group_end < len(active) and active[group_end].kernel_names
                else None
            )
            group_entries: list[tuple[int, tuple[str, ...] | None]] = []
            scan_idx = ca_idx
            while scan_idx < len(callargs_full):
                k, fp = callargs_full[scan_idx]
                if next_first is not None and k == next_first:
                    break
                if k == first_kernel:
                    group_entries.append((scan_idx, fp))
                scan_idx += 1

            if group_entries:
                self._assign_group_slots(active[i:group_end], group_entries)
            ca_idx = scan_idx
            i = group_end

        # Count assigned slots; injected helper kernels (MemCpy, etc.) are excluded.
        self._coverable_slot_count = len(
            {
                idx
                for sg in active
                if sg.location >= 0 and sg.kernels > 0
                for idx in range(sg.location, sg.location + sg.kernels)
            }
        )

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

        Pass-through subgraphs (input name == output name) are always chained after
        their predecessor — they are injected by the Neutron converter for data routing
        and always connect to exactly one producer.
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
        Neutron-internal consumer of sg_i's outputs. Subgraphs not part of a longer
        chain form singletons. Subgraphs with multiple predecessors (join targets)
        or multiple successors (fork sources) break the chain.
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

        # Any subgraphs unreachable from a chain root (e.g. fork targets) become singletons.
        for sg in active:
            if sg.num not in visited:
                chains.append([sg])

        return chains

    def _get_chain_boundary_io(
        self, chain: list[SubgraphInfo]
    ) -> tuple[list[str], list[str]]:
        """Return the external (boundary) inputs and outputs of a Neutron subgraph chain.

        Chain inputs  = dynamic inputs of the first subgraph in the chain.
        Chain outputs = dynamic outputs not consumed internally by any later subgraph.

        :param chain: Ordered list of SubgraphInfo forming one execution chain.
        :return: (chain_inputs, chain_outputs) as lists of tensor names.
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
            and count_tensor_matches(tf_node.inputs, sg_inputs) == len(tf_node.inputs)
        )

    def _outputs_match(self, sg_outputs: list[str], chain_outputs: list[str]) -> bool:
        """Return True if all chain_outputs are covered by sg_outputs."""
        return (
            bool(chain_outputs)
            and bool(sg_outputs)
            and count_tensor_matches(chain_outputs, sg_outputs) == len(chain_outputs)
        )

    def _find_matching_tflite_chain(
        self, sg_inputs: list[str], sg_outputs: list[str]
    ) -> list[int]:
        """Find the TFLite operator chain whose collective I/O matches the given boundary I/O.

        Algorithm:
          1. Identify candidate starting nodes whose dynamic inputs match sg_inputs.
          2. For each candidate, check for a 1-to-1 output match.
          3. Otherwise, extend the chain forward greedily until the outputs match sg_outputs.

        When all sg_outputs are Neutron-internal (not in the TFLite tensor name set),
        the converter renamed the final output tensor (e.g. "newOut" in MobileNetV1
        GlobalAvgPool or "newOut" in batch>1 FullyConnected). In that case
        **input-only matching** is used: return the single TFLite node whose inputs
        match sg_inputs without any extension. This is safe for both intermediate and
        terminal Neutron chains because it never absorbs TFLite nodes that belong to
        later Neutron chains.

        :param sg_inputs: Dynamic inputs of the Neutron subgraph / chain boundary.
        :param sg_outputs: Dynamic outputs of the Neutron subgraph / chain boundary.
        :return: Ordered list of TFLite node locations forming the match, or [].
        """
        output_is_tflite = any(o in self._tflite_tensor_names for o in sg_outputs)

        if not output_is_tflite:
            # The converter renamed all outputs of this chain (e.g. "newOut").
            # Match by inputs only: return the first TFLite node whose inputs match,
            # without extension. Extending would wrongly absorb successor TFLite
            # operators that belong to later Neutron chains.
            for start in (
                n for n in self.tflite_nodes if self._inputs_match(sg_inputs, n)
            ):
                return [start.location]
            return []

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

        :param chain_locs: Locations of nodes already in the chain.
        :param chain_outputs: Current cumulative outputs of the chain.
        :param node_by_loc: Location -> Node lookup map.
        :param input_to_locs: Tensor name -> list of consumer locations map.
        :return: Location of the unique eligible next node, or None.
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
        """Map TFLite node locations to Neutron kernel CALLARGS indices.

        Neutron subgraphs are first grouped into chains; each chain is matched as a
        unit against a TFLite operator chain using dynamic I/O tensor names.

        When the microcode header is authoritative (batch > 1 logs) a completeness
        check verifies that every coverable CALLARGS slot is covered. Injected helper
        slots (MemCpy, etc.) that have no TFLite counterpart are excluded from the
        requirement via _coverable_slot_count. If the check fails the method returns
        an empty map to avoid silently producing wrong profiling data.

        :return: Dict: TFLite node location -> tuple of Neutron kernel CALLARGS indices.
        """
        result: dict[int, set[int]] = {}

        chains = self._get_neutron_subgraph_chains()

        for chain in chains:
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

        # Sanity check (batch-aware logs only): every coverable CALLARGS slot must be
        # mapped. A gap means the remapping failed and the partial map would produce
        # wrong profiling data. For batch=1 logs (no microcode header override) partial
        # coverage is expected for fork/join topologies and no check is applied.
        if self._microcode_kernels_authoritative:
            required = self._coverable_slot_count or self.neutron_kernels_num
            mapped = {idx for v in self.tflite_to_neutron_map.values() for idx in v}
            if len(mapped) < required:
                logging.info(
                    f"NeutronMap: {len(mapped)}/{required} coverable slots mapped "
                    f"(neutron_kernels_num={self.neutron_kernels_num}). Returning empty map."
                )
                self.tflite_to_neutron_map = {}

        return self.tflite_to_neutron_map

    def get_edge_to_neutron_map(self) -> dict[int, tuple[int, ...]]:
        """Map Edge node handles to Neutron kernel CALLARGS indices.

        Calls get_tflite_to_neutron_map() if not already computed, then composes
        it with the edge_to_tflite_map supplied at construction time.

        :return: Dict: Edge handle -> tuple of Neutron kernel CALLARGS indices.
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

        Every CALLARGS slot index from 0 to neutron_kernels_num (inclusive) is
        present in the result. Slots with no corresponding Edge operator map to an
        empty tuple. One extra entry (at index neutron_kernels_num) covers the
        Neutron Dump event emitted at the end of each inference.

        :return: Dict: Neutron kernel CALLARGS index -> tuple of Edge handles.
                 All indices up to neutron_kernels_num are present (empty tuple if unmapped).
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
