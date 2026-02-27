import logging
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Tuple

import torch
import torch.fx.passes.operator_support as ops
from torch.fx._compatibility import compatibility
from torch.fx.node import Target
from torch.fx.passes.splitter_base import (
    _SplitterBase,
    _SplitterSettingBase,
)
from torch.fx.passes.tools_common import (
    CALLABLE_NODE_OPS,
    FxNetAccFusionsFinder,
    NodeList,
    NodeSet,
    is_node_output_tensor,
)
from torch_tensorrt.dynamo._defaults import (
    MIN_BLOCK_SIZE,
    REQUIRE_FULL_COMPILATION,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
)

logger = logging.getLogger(__name__)

NON_COMPUTE_NODES = {"torch.ops.aten.view", "_operator.getitem"}
NON_ACC_BACKEND_NAME = "None"


@compatibility(is_backward_compatible=False)
@dataclass
class Subgraph:
    is_acc: bool
    backend: str
    nodes: NodeList
    device_ordinal: Optional[int] = None


class BackendOpSupportTester(ops.OperatorSupportBase):  # type: ignore
    """Class to determine whether operators are supported by specific backends"""

    def __init__(
        self,
        backend_support_map: Dict[str, Collection[Target]],
        backend_priority: List[str],
        torch_executed_ops: Collection[Target] = set(),
    ) -> None:
        super().__init__()

        # Initialize sets of supported/unsupported operators
        self.supported_operators: Dict[str, int] = {}
        self.unsupported_operators: Dict[str, int] = {}
        self.torch_executed_ops = torch_executed_ops
        # Map of backend names to sets of supported operators
        self.backend_support_map = backend_support_map
        # Ordered list of backend names, from highest to lowest priority
        self.backend_priority = backend_priority

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> Tuple[bool, str]:
        node_name = ConverterRegistry.qualified_name_or_str(node.target)

        for i, backend_name in enumerate(self.backend_priority):
            supported_ops = self.backend_support_map.get(backend_name, set())
            supported_ops = {
                ConverterRegistry.qualified_name_or_str(op) for op in supported_ops
            }

            if (
                (node_name in supported_ops or node.op == "get_attr")
                and node_name not in self.torch_executed_ops
                and node.target not in self.torch_executed_ops
            ):
                # If node is a proper, supported computational node, store the operator
                if not node.is_impure() and node.op != "get_attr":
                    if node_name not in self.supported_operators:
                        self.supported_operators[f"{backend_name}_{node_name}"] = 1
                    else:
                        self.supported_operators[f"{backend_name}_{node_name}"] += 1

                return True, backend_name
            else:
                if i == len(self.backend_priority) - 1 and not node.is_impure():
                    if node_name not in self.unsupported_operators:
                        self.unsupported_operators[node_name] = 1
                    else:
                        self.unsupported_operators[node_name] += 1

        return False, NON_ACC_BACKEND_NAME

    def print_support_overview(self, num_acc_subgraphs: Optional[int] = None) -> None:
        if num_acc_subgraphs is not None:
            logger.debug(
                f"\nNumber of Accelerated Subgraphs Generated: {num_acc_subgraphs}"
            )

        # Reformat support messages for debugger to print node overview as a single string
        supported_nodes_str = "\nSupported Nodes:\n"
        for node_name, count in self.supported_operators.items():
            supported_nodes_str += f"- {node_name} + Operator Count: {count}\n"

        logger.debug(supported_nodes_str)

        if self.unsupported_operators:
            unsupported_nodes_str = "\nUnsupported or Excluded Nodes:\n"
            for node_name, count in self.unsupported_operators.items():
                unsupported_nodes_str += f"- {node_name} + Operator Count: {count}\n"

            logger.debug(unsupported_nodes_str)
        else:
            logger.debug("\nAll Nodes Supported\n")


class HierarchicalAdjacencyPartitioner(_SplitterBase):  # type: ignore
    """Hierarchical Adjacency Partitioner to split an FX graph into subgraphs based on backend priority

    This partitioner extends the TRTPartitioner of adjacency_partitioner with backend priority awareness,
    allowing different parts of the model to be executed on different backends based on
    operator support and priority ordering.

    Args:
        module: FX GraphModule to partition
        operator_support: OperatorSupport class describing allowed operators
        backend_support_map: Dictionary mapping backend names to sets of supported operators
        backend_priority: Ordered list of backend names, from highest to lowest priority
        allowed_single_node_partition_ops: Nodes which can be included in single-node partitions
        min_block_size: Minimum number of computational operators per block
        require_full_compilation: Require that all computational operators be run in TRT
    Returns:
        torch.fx.GraphModule
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: ops.OperatorSupportBase,
        backend_support_map: Dict[str, Collection[Target]],
        backend_priority: List[str],
        allowed_single_node_partition_ops: Optional[Collection[str]] = None,
        min_block_size: int = MIN_BLOCK_SIZE,
        require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
        return_tuple: bool = False,
        skip_fusion: bool = False,
    ):
        """
        Preprocesses graph before splitting with backend priority awareness
        """
        assert isinstance(module, torch.fx.GraphModule)

        self.module = module
        self.backend_support_map = backend_support_map
        self.backend_priority = backend_priority

        self.settings = _SplitterSettingBase(
            min_acc_module_size=min_block_size,
            allow_non_tensor=True,
            skip_fusion=skip_fusion,
        )
        self.operator_support = operator_support

        # Get all accelerated nodes based on operator support conditions
        self.acc_nodes = FxNetAccNodesFinder(
            self.module, self.operator_support, self.settings.allow_non_tensor
        )()

        if self.settings.skip_fusion:
            self.fusions = {}
        else:
            self.fusions = FxNetAccFusionsFinder(module, set(self.acc_nodes))()

        # Modify deps to add more deps for fused nodes
        self.deps = self.find_deps()
        self.update_deps_for_fusions()

        self.non_acc_submodule_name = "_run_on_gpu_"
        self._node_submodule_map: Dict[str, str] = {}

        self.num_accelerated_subgraphs: Optional[int] = None
        self.allowed_single_node_partition_ops = allowed_single_node_partition_ops
        self.require_full_compilation = require_full_compilation
        self._return_tuple = return_tuple

    def remove_small_acc_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent GPU submodules.
        """
        result: List[Subgraph] = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                if (
                    len(subgraph.nodes) >= self.settings.min_acc_module_size
                    or self.require_full_compilation
                    or (
                        self.allowed_single_node_partition_ops is not None
                        and any(
                            ConverterRegistry.qualified_name_or_str(node.target)
                            in self.allowed_single_node_partition_ops
                            for node in subgraph.nodes
                        )
                    )
                ):
                    result.append(subgraph)
                else:
                    logger.debug(
                        "Eliminating acc subgraph because it's smaller than the threshold: "
                        f"{len(subgraph.nodes)} < {self.settings.min_acc_module_size}"
                    )
                    # if the last subgraph result[-1] is non-acc or has the same backend, merge the current subgraph into it
                    if result and (
                        not result[-1].is_acc or result[-1].backend == subgraph.backend
                    ):
                        result[-1].nodes.extend(subgraph.nodes)
                    else:
                        # if the last subgraph result[-1] has different backends, then append the current subgraph as non-acc
                        subgraph.is_acc = False
                        subgraph.backend = "None"
                        result.append(subgraph)
            else:
                if result and not result[-1].is_acc:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    if result:
                        if result[-1].backend == subgraph.backend:
                            result[-1].nodes.extend(subgraph.nodes)
                        else:
                            result.append(subgraph)
                    else:
                        result.append(subgraph)
        return result

    def partition_graph(self) -> torch.fx.GraphModule:
        """Partitions the GraphModule into subgraphs based on backend priority

        Returns a GraphModule with submodules for each segment
        """
        # Delegate nodes based on operator coverage
        subgraphs = self.put_nodes_into_subgraphs()

        # A graph is fully supported if there is a single partition and all operators are supported/convertible
        full_support = len([s for s in subgraphs if s.is_acc]) == 1 and not getattr(
            self.operator_support, "unsupported_operators", True
        )

        if not full_support and self.require_full_compilation:
            raise AssertionError(
                "require_full_compilation=True was specified, but model is not fully supported or multiple partitions are found"
            )

        if (
            full_support
            and self.require_full_compilation
            and self.settings.min_acc_module_size != MIN_BLOCK_SIZE
        ):
            logger.warning(
                "Detected both require_full_compilation and min_block_size compilation "
                "arguments were specified. Disregarding min_block_size argument for "
                "fully supported model."
            )

        # Remove segments smaller than the block size (with exceptions)
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)

        # Set the number of accelerated subgraphs to be generated
        self.num_accelerated_subgraphs = len([s for s in subgraphs if s.is_acc])

        # Tag the accelerated nodes and split the graph accordingly
        self.tag(subgraphs)
        return self.split()

    def starter_nodes(self) -> Tuple[NodeSet, NodeSet]:
        """Generates starter nodes for partitioning + segmentation"""
        # Starter accelerated nodes are all callable accelerated ops
        starter_acc_nodes = {
            node for node in self.acc_nodes if node.op in CALLABLE_NODE_OPS
        }

        # Started non-accelerated nodes are the rest of the callable nodes
        starter_non_acc_nodes = {
            node
            for node in self.module.graph.nodes
            if (node not in starter_acc_nodes and node.op in CALLABLE_NODE_OPS)
        }
        return starter_non_acc_nodes, starter_acc_nodes

    def put_nodes_into_subgraphs(self) -> list[Subgraph]:
        # We start graph traversal from leaf nodes
        current_cpu_nodes, current_acc_nodes = self.starter_nodes()
        visited_nodes: NodeSet = set()

        # Determine which subgraph to start from based on which subgraph has
        # 0-dep node
        acc_subgraph: bool = not any(len(self.deps[n]) == 0 for n in current_cpu_nodes)

        current_subgraph_nodes: NodeList = []

        # Result accumulator
        subgraphs: list[Subgraph] = []
        while current_cpu_nodes or current_acc_nodes:
            # Find the first node that should belong to the current subgraph and has all dependencies resolved
            current_nodes = current_acc_nodes if acc_subgraph else current_cpu_nodes
            node = next(
                (n for n in current_nodes if self.deps[n] <= visited_nodes),
                None,
            )

            # If no node was found, then it's time to flip the mode and start a new subgraph
            if node is None:
                if not current_subgraph_nodes:
                    raise FxNetSplitterInternalError("Subgraph can't be empty")

                subgraphs.append(
                    Subgraph(
                        is_acc=acc_subgraph,
                        nodes=current_subgraph_nodes,
                        backend=(
                            current_subgraph_nodes[-1].backend
                            if acc_subgraph
                            else "None"
                        ),
                    )
                )
                acc_subgraph = not acc_subgraph
                current_subgraph_nodes = []
                continue

            # If the backend changed, then it's time to start a new subgraph
            if (
                current_subgraph_nodes
                and current_subgraph_nodes[-1].backend != node.backend
            ):
                if not current_subgraph_nodes:
                    raise FxNetSplitterInternalError("Subgraph can't be empty")

                subgraphs.append(
                    Subgraph(
                        is_acc=acc_subgraph,
                        nodes=current_subgraph_nodes,
                        backend=current_subgraph_nodes[-1].backend,
                    )
                )
                current_subgraph_nodes = []
                continue

            current_nodes.remove(node)
            visited_nodes.add(node)
            current_subgraph_nodes.append(node)

            # Add fusion buddies
            if node in self.fusions:
                if node in self.acc_nodes:
                    current_acc_nodes.update(self.fusions[node] - visited_nodes)
                else:
                    current_cpu_nodes.update(self.fusions[node] - visited_nodes)

            # Put depending nodes into the queue
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                # Add downstream nodes
                if user in self.acc_nodes:
                    current_acc_nodes.add(user)
                else:
                    current_cpu_nodes.add(user)

        # Check if the last subgraph was not created
        if current_subgraph_nodes:
            subgraphs.append(
                Subgraph(
                    is_acc=acc_subgraph,
                    nodes=current_subgraph_nodes,
                    backend=(
                        current_subgraph_nodes[-1].backend if acc_subgraph else "None"
                    ),
                )
            )

        if not subgraphs:
            raise FxNetSplitterInternalError("Couldn't create subgraphs")

        return subgraphs

    def tag(self, subgraphs: list[Subgraph]) -> None:
        self.tags: list[str] = []
        for subgraph in subgraphs:
            tag = (
                f"_run_on_acc_{subgraph.backend}_{len(self.tags)}"
                if subgraph.is_acc
                else f"{self.non_acc_submodule_name}{len(self.tags)}"
            )
            self.tags.append(tag)
            for node in subgraph.nodes:
                if hasattr(node, "tag"):
                    raise FxNetSplitterInternalError(f"Node {node} was already tagged")

                node.tag = tag
                self._node_submodule_map[node.name] = tag


@compatibility(is_backward_compatible=False)
class FxNetAccNodesFinder:
    """
    Finds a set of nodes that can be supported on ACC, excluding nodes that have non-tensor
    input/output to cpu nodes to prevent non-tensor data flow between backends and cpu.

    I.e. if we have a chain:

    ACC_NODE_1 -> ACC_NODE_2 -> ACC_NODE_3 -> CPU_NODE_1

    where every ACC node produces non-tensor output, then they all should be treated as CPU nodes.

    This behavior can be turned off by passing allow_non_tensor=True.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: ops.OperatorSupportBase,
        allow_non_tensor: bool,
    ):
        self.module = module
        self.operator_support = operator_support
        self.allow_non_tensor = allow_non_tensor
        self.acc_nodes: NodeSet = set()

    def reduce_acc_nodes_non_tensor_input_helper(self, cpu_worklist: NodeList) -> None:
        """
        Transitively excludes nodes from ACC supported set.
        For every node in the worklist:
        - removes its downstream ACC nodes from ACC supported set,
        - if any downstream ACC node produces non-tensor output,
          then it gets added into the worklist.
        """
        while cpu_worklist:
            node = cpu_worklist.pop(0)

            for user in node.users:
                if user in self.acc_nodes:
                    self.acc_nodes.remove(user)
                    if not is_node_output_tensor(user):
                        cpu_worklist.append(user)

    def reduce_acc_nodes_non_tensor_input(self) -> None:
        """
        Excludes nodes from ACC supported set that have direct
        upstream CPU nodes that produce non-tensor outputs.
        """
        non_tensor_cpu_nodes: NodeList = []

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if node in self.acc_nodes:
                continue
            if is_node_output_tensor(node):
                continue
            non_tensor_cpu_nodes.append(node)

        self.reduce_acc_nodes_non_tensor_input_helper(non_tensor_cpu_nodes)

    def reduce_acc_nodes_non_tensor_output(self) -> None:
        """
        Excludes nodes from ACC supported set that produce non-tensor
        outputs and have downstream CPU nodes.
        """
        while True:
            new_cpu_nodes: NodeList = []

            for acc_node in self.acc_nodes:
                if is_node_output_tensor(acc_node):
                    continue
                for user in acc_node.users:
                    if user not in self.acc_nodes:
                        new_cpu_nodes.append(acc_node)
                        break

            if not new_cpu_nodes:
                break

            for new_cpu_node in new_cpu_nodes:
                self.acc_nodes.remove(new_cpu_node)

            self.reduce_acc_nodes_non_tensor_input_helper(new_cpu_nodes)

    def __call__(self) -> NodeSet:
        submodules = dict(self.module.named_modules())
        backend = NON_ACC_BACKEND_NAME
        for n in self.module.graph.nodes:
            # Group non-compute nodes with previous compute nodes
            if ConverterRegistry.qualified_name_or_str(n.target) in NON_COMPUTE_NODES:
                n.backend = backend
                if backend != NON_ACC_BACKEND_NAME:
                    self.acc_nodes.add(n)
                continue

            if n.op in CALLABLE_NODE_OPS:
                is_supported, backend = self.operator_support.is_node_supported(
                    submodules, n
                )
                if is_supported:
                    n.backend = backend
                    self.acc_nodes.add(n)
                else:
                    n.backend = NON_ACC_BACKEND_NAME

        if not self.allow_non_tensor:
            self.reduce_acc_nodes_non_tensor_input()
            self.reduce_acc_nodes_non_tensor_output()

        return self.acc_nodes


@compatibility(is_backward_compatible=False)
class FxNetSplitterInternalError(Exception):
    pass


def hierarchical_adjacency_partition(
    gm: torch.fx.GraphModule,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Collection[Target] = set(),
    backend_support_map: Optional[Dict[str, Collection[Target]]] = None,
    backend_priority: Optional[List[str]] = None,
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
    skip_fusion: bool = False,
) -> Tuple[torch.fx.GraphModule, BackendOpSupportTester]:
    """Partition an FX GraphModule with aten ops into submodules using hierarchical partitioning
    based on backend priority and operator support

    Args:
        gm: FX GraphModule to partition
        min_block_size: Minimum number of operators per TRT-Engine Block
        backend_support_map: Dictionary mapping backend names to sets of supported operators
        backend_priority: Ordered list of backend names, from highest to lowest priority
        require_full_compilation: Require that all computational operators be run in TRT
        skip_fusion: Skip fusions found by FxNetAccFusionsFinder
    Returns:
        torch.fx.GraphModule, BackendOpSupportTester
    """
    # Ensure graph is clean prior to partitioning
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    # Default backend support map if none provided
    if backend_support_map is None:
        backend_support_map = {
            "tensorrt": CONVERTERS.keys(),
            "inductor": set(),
        }

    # Default backend priority if none provided
    if backend_priority is None:
        backend_priority = ["tensorrt", "inductor"]

    # Construct BackendOpSupportTester
    supported_ops = BackendOpSupportTester(
        backend_support_map=backend_support_map,
        backend_priority=backend_priority,
        torch_executed_ops=torch_executed_ops,
    )
    partitioner = HierarchicalAdjacencyPartitioner(
        gm,
        supported_ops,
        backend_support_map=backend_support_map,
        backend_priority=backend_priority,
        min_block_size=min_block_size,
        require_full_compilation=require_full_compilation,
        skip_fusion=skip_fusion,
    )

    partitioned_graph = partitioner.partition_graph()

    supported_ops.print_support_overview(partitioner.num_accelerated_subgraphs)

    return partitioned_graph, supported_ops
