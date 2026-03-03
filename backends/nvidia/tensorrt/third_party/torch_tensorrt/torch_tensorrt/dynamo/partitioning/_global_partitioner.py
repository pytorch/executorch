import logging
from typing import Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Target
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.operator_support import OperatorSupport, SupportDict
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


class TRTPartitioner(CapabilityBasedPartitioner):  # type: ignore[misc]
    """Partitioner to split an FX graph into subgraphs based on operator support

    Args:
        graph_module: FX GraphModule to partition
        operator_support: OperatorSupport class describing allowed operators
        non_compute_ops: Operators which are not considered computational (e.g. getattr)
        allowed_single_node_partition_ops: Nodes which can be included in single-node partitions.
            Generally useful for module-level exclusion ops which are intensive despite being single functions
        min_block_size: Minimum number of computational operators per block
        require_full_compilation: Require that all computational operators be run in TRT
    Returns:
        torch.fx.GraphModule
    """

    def __init__(
        self,
        graph_module: GraphModule,
        operator_support: OperatorSupport,
        *,
        non_compute_ops: Optional[Sequence[str]] = None,
        allowed_single_node_partition_ops: Optional[Collection[str]] = None,
        min_block_size: int = MIN_BLOCK_SIZE,
        require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
    ) -> None:
        super().__init__(
            graph_module,
            operator_support,
            allows_single_node_partition=True,
            non_compute_ops=non_compute_ops,
            allowed_single_node_partition_ops=allowed_single_node_partition_ops,
        )

        self.min_block_size = min_block_size
        self.require_full_compilation = require_full_compilation

    def propose_partitions(self) -> List[Partition]:
        # Propose partitions using the default, then refine the results
        initial_proposed_partitions = super().propose_partitions()
        partitions = dict(enumerate(initial_proposed_partitions))

        # A graph is fully supported if there is a single partition and all operators are supported/convertible
        full_support = len(partitions) == 1 and not getattr(
            self.operator_support, "unsupported_operators", True
        )

        if not full_support and self.require_full_compilation:
            raise AssertionError(
                "require_full_compilation=True was specified, but model is not fully supported"
            )

        if (
            full_support
            and self.require_full_compilation
            and self.min_block_size != MIN_BLOCK_SIZE
        ):
            logger.warning(
                "Detected both require_full_compilation and min_block_size compilation "
                "arguments were specified. Disregarding min_block_size argument for "
                "fully supported model."
            )

        # For each partition, determine whether or not the number of computational operators
        # exceeds the threshold, and if not, remove that partition
        partitions_to_remove = {}
        for id, partition in partitions.items():
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            exempted_partition = False

            compute_node_count = 0
            for node in partition.nodes:
                # Partitions are exempted from min_block_size if they contain an allowed single-node op
                if node.op == "call_function" and (
                    self.allowed_single_node_partition_ops is not None
                    and ConverterRegistry.qualified_name_or_str(node.target)
                    in self.allowed_single_node_partition_ops
                ):
                    exempted_partition = True
                    break
                elif (
                    node.op == "call_function"
                    and ConverterRegistry.qualified_name_or_str(node.target)
                    not in non_compute_ops
                ):
                    compute_node_count += 1

            if (
                compute_node_count < self.min_block_size
                and not exempted_partition
                and not (full_support and self.require_full_compilation)
            ):
                partitions_to_remove[id] = compute_node_count

        # Remove any nodes violating the criteria specified by the user
        for id, count in partitions_to_remove.items():
            logger.debug(
                f"Removing partition which has {count} < {self.min_block_size} computational operators"
            )
            del partitions[id]

        return [partitions[k] for k in sorted(partitions.keys())]

    def partition_and_fuse(self) -> GraphModule:
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions)
        return fused_gm


class TorchTensorRTOperatorSupport(OperatorSupport):  # type: ignore[misc]
    """Class to determine whether operators within a module are supported"""

    def __init__(
        self,
        support_dict: Optional[SupportDict] = None,
        torch_executed_ops: Collection[Target] = set(),
    ):
        super().__init__(support_dict)

        # Initialize sets of supported/unsupported operators
        self.supported_operators: Dict[str, int] = {}
        self.unsupported_operators: Dict[str, int] = {}
        self.torch_executed_ops: Collection[Target] = torch_executed_ops

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        node_name = ConverterRegistry.qualified_name_or_str(node.target)

        if (
            (node in CONVERTERS or node.op == "get_attr")
            and node_name not in self.torch_executed_ops
            and node.target not in self.torch_executed_ops
        ):
            # If node is a proper, supported computational node, store the operator
            if not node.is_impure() and node.op != "get_attr":
                if node_name not in self.supported_operators:
                    self.supported_operators[node_name] = 1
                else:
                    self.supported_operators[node_name] += 1

            return True
        else:
            if not node.is_impure():
                if node_name not in self.unsupported_operators:
                    self.unsupported_operators[node_name] = 1
                else:
                    self.unsupported_operators[node_name] += 1

            return False

    def print_support_overview(
        self, num_trt_blocks: Optional[int] = None, print_node_support: bool = False
    ) -> None:
        if num_trt_blocks is not None:
            logger.debug(
                f"\nNumber of TensorRT-Accelerated Engines Generated: {num_trt_blocks}"
            )

        if print_node_support:
            # Reformat support messages for debugger to print node overview as a single string
            supported_nodes_str = "\nSupported Nodes:\n"
            for node_name, count in self.supported_operators.items():
                supported_nodes_str += f"- {node_name} + Operator Count: {count}\n"

            logger.debug(supported_nodes_str)

            if self.unsupported_operators:
                unsupported_nodes_str = "\nUnsupported or Excluded Nodes:\n"
                for node_name, count in self.unsupported_operators.items():
                    unsupported_nodes_str += (
                        f"- {node_name} + Operator Count: {count}\n"
                    )

                logger.debug(unsupported_nodes_str)
            else:
                logger.debug("\nAll Nodes Supported\n")


def partition(
    gm: torch.fx.GraphModule,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Collection[Target] = set(),
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
) -> Tuple[torch.fx.GraphModule, TorchTensorRTOperatorSupport]:
    """Partition an FX GraphModule with aten ops into TRT engines
    Partitioning is based on converter operator support

    Args:
        gm: FX GraphModule to partition
        min_block_size: Minimum number of operators per TRT-Engine Block
        torch_executed_ops: Collection of operations to run in Torch, regardless of converter coverage
        require_full_compilation: Whether to require that all operators be run in TRT
    Returns:
        torch.fx.GraphModule, TorchTensorRTOperatorSupport
    """
    supported_ops = TorchTensorRTOperatorSupport(torch_executed_ops=torch_executed_ops)
    partitioner = TRTPartitioner(
        gm,
        supported_ops,
        min_block_size=min_block_size,
        require_full_compilation=require_full_compilation,
    )

    # Determine partitions based on user specifications and operator support
    # Then, fuse partitions and display overview of supported/unsupported operators
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions, prefix="_run_on_acc_")
    supported_ops.print_support_overview(len(partitions))

    return fused_graph, supported_ops
