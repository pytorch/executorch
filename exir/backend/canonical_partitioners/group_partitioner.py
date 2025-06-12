# mypy: allow-untyped-defs
import collections
import itertools
import logging
from collections.abc import Sequence
from typing import List, Optional

from torch.fx.graph_module import GraphModule
from torch.fx.node import _get_qualified_name, Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.operator_support import OperatorSupportBase


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class _DependencyViewer:
    def __init__(self, graph_module: GraphModule):
        self.downstreams = collections.defaultdict(set)
        self.upstreams = collections.defaultdict(set)

        for node in reversed(graph_module.graph.nodes):
            for output_node in node.users:
                # add output_node and output_node's downstream dependency
                self.downstreams[node].add(output_node)
                self.downstreams[node].update(self.downstreams[output_node])

        for node in graph_module.graph.nodes:
            for input_node in node.all_input_nodes:
                self.upstreams[node].add(input_node)
                self.upstreams[node].update(self.upstreams[input_node])

    def downstreams_of(self, node: Node) -> set[Node]:
        return self.downstreams[node]

    def upstreams_of(self, node: Node) -> set[Node]:
        return self.upstreams[node]


class GroupBasedPartitioner(CapabilityBasedPartitioner):
    """
    A specialized partitioner that extends the CapabilityBasedPartitioner from PyTorch FX.

    GroupBasedPartitioner allows for explicit grouping of nodes into partitions based on
    predefined node groups, while also supporting automatic partitioning for nodes not
    included in any group. Nodes are only allowed to be in one group.

    Features:
    - Explicit Node Grouping: Allows specifying groups of nodes that should be kept together
      in the same partition.
    - Automatic Partitioning: Nodes not included in any explicit group are automatically
      partitioned based on operator support.
    - Cycle Prevention: Ensures that partitioning doesn't create cycles in the execution graph.
    - Single Node Partition Control: Options to allow or disallow single-node partitions,
      with exceptions for specific operations.

    Args:
        graph_module: The FX GraphModule to be partitioned.
        operator_support: Interface to determine if a node is supported by the target backend.
        allows_single_node_partition: Whether to allow single-node partitions. Default: False.
        non_compute_ops: Operations not counted for single-node partition determination. Default: None.
        allowed_single_node_partition_ops: Operations allowed as single-node partitions. Default: None.
        node_groups: Lists of nodes to group together in partitions. Default: None.
    """
    def __init__(
        self,
        graph_module: GraphModule,
        operator_support: OperatorSupportBase,
        allows_single_node_partition: bool = False,
        non_compute_ops: Optional[Sequence[str]] = None,
        allowed_single_node_partition_ops: Optional[Sequence[str]] = None,
        node_groups: List[List[Node]] = None,
    ) -> None:
        super().__init__(
            graph_module=graph_module,
            operator_support=operator_support,
            allows_single_node_partition=allows_single_node_partition,
            non_compute_ops=non_compute_ops,
            allowed_single_node_partition_ops=allowed_single_node_partition_ops,
        )
        self.dependency_viewer = _DependencyViewer(graph_module)
        self.node_groups = (
            [set(node_group) for node_group in node_groups] if node_groups else None
        )
        self.node_to_group = collections.defaultdict(int)
        self.all_nodes_in_groups = set()
        if node_groups:
            for i, group in enumerate(self.node_groups):
                for node in group:
                    # Node is in multiple groups - not allowed
                    if node in self.node_to_group:
                        raise ValueError(f"Node {node} exists in multiple groups.")
                    self.node_to_group[node] = i
                    self.all_nodes_in_groups.add(node)

    def propose_partitions(self) -> list[Partition]:
        # partition_map is a mapping from partition id to a set of partition id's.
        # The value set contains all the partition ids that can be reached by doing a
        # DFS starting from the partition id in the key.
        partition_map: dict[int, set] = collections.defaultdict(set)

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: dict[Node, int] = {}  # mapping from node to partition_id
        partitions_by_id: dict[int, Partition] = (
            {}
        )  # mapping from partition_id to partition
        nodes_order: dict[Node, int] = (
            {}
        )  # mapping from nodes to reversed topological order
        partitions_order: dict[int, int] = (
            {}
        )  # mapping from partition_id to minimum topo order of nodes in partition
        partition_users: dict[int, set] = (
            {}
        )  # mapping from partition_id to partition users
        new_partition_id = itertools.count()

        group_to_partition_id = {}  # mapping from group id to partition id

        # Try to merge partitions that don't create cycles
        def can_merge(p1, p2):
            # Check if merging would create a cycle
            p1_nodes = set(partitions_by_id[p1].nodes.keys())
            p2_nodes = set(partitions_by_id[p2].nodes.keys())

            # Create a combined set of nodes from both partitions
            combined_nodes = p1_nodes.union(p2_nodes)

            # For each node in the combined partition, check if any of its external downstream nodes
            # have downstream nodes that are in the combined partition
            for node in combined_nodes:
                # Get all downstream nodes that are not in the combined partition
                external_downstreams = {
                    n
                    for n in self.dependency_viewer.downstreams_of(node)
                    if n not in combined_nodes
                }
                # Check if any of these external downstream nodes have downstream nodes that are in the combined partition
                for external_node in external_downstreams:
                    for downstream_node in self.dependency_viewer.downstreams_of(
                        external_node
                    ):
                        if downstream_node in combined_nodes:
                            return False

            return True

        # Preprocess nodes to put them in same partition
        if self.node_groups:
            for i, group in enumerate(self.node_groups):
                # Create a partition for each group
                partition_id = next(new_partition_id)
                partition = Partition(id=partition_id, nodes=set())
                partitions_by_id[partition_id] = partition
                partitions_order[partition_id] = partition_id
                group_to_partition_id[i] = partition_id

                # Add all supported nodes from the group to the partition
                for node in group:
                    if self._is_node_supported(node):
                        partition.add_node(node)
                        assignment[node] = partition_id
                        nodes_order[node] = partition_id

                # Set partition users
                partition_users[partition_id] = {
                    user
                    for node in partition.nodes
                    for user in node.users
                    if user not in partition.nodes
                }

                # Update partition map
                for node in partition.nodes:
                    for user in node.users:
                        target_id = assignment.get(user)
                        if target_id is not None and target_id != partition_id:
                            partition_map[partition_id].add(target_id)
                            partition_map[partition_id].update(partition_map[target_id])

        # Process remaining nodes
        for node in reversed(self.graph_module.graph.nodes):
            if node in assignment or not self._is_node_supported(node):
                continue

            partition_id = next(new_partition_id)
            nodes_order[node] = partition_id
            partitions_order[partition_id] = partition_id
            partitions_by_id[partition_id] = Partition(id=partition_id, nodes=[node])
            assignment[node] = partition_id
            partition_users[partition_id] = set(node.users)

            # Update partition map
            for user in node.users:
                target_id = assignment.get(user)
                if target_id is not None:
                    partition_map[partition_id].add(target_id)
                    partition_map[partition_id].update(partition_map[target_id])

        # Merge partitions when possible
        merged = True
        while merged:
            merged = False
            partition_ids = list(partitions_by_id.keys())
            for i, p1 in enumerate(partition_ids):
                if p1 not in partitions_by_id:
                    continue

                for p2 in partition_ids[i + 1 :]:
                    if p2 not in partitions_by_id:
                        continue

                    # Try to merge partitions if it doesn't create cycles
                    if can_merge(p1, p2):
                        # Merge p2 into p1
                        partitions_by_id[p1].nodes.update(partitions_by_id[p2].nodes)
                        for node in partitions_by_id[p2].nodes:
                            assignment[node] = p1

                        # Update partition users
                        all_users = partition_users[p1] | partition_users[p2]
                        all_users.difference_update(partitions_by_id[p1].nodes)
                        partition_users[p1] = all_users

                        # Update partition map
                        partition_map[p1].update(partition_map[p2])

                        # Update partition order
                        partitions_order[p1] = min(
                            partitions_order[p1], partitions_order[p2]
                        )

                        # Remove p2
                        del partitions_by_id[p2]
                        del partition_users[p2]
                        del partitions_order[p2]
                        if p2 in partition_map:
                            del partition_map[p2]

                        merged = True
                        break

                if merged:
                    break

        # Post-processing for getitem nodes
        nodes_reassignment = {}
        for node in self.graph_module.graph.nodes:
            is_tuple_output = True
            for user in node.users:
                if (
                    user.op != "call_function"
                    or _get_qualified_name(user.target) != "_operator.getitem"
                ):
                    is_tuple_output = False
                    break

            # node has tuple outputs, re-assign all following getitem node into node's partition
            if is_tuple_output:
                id = assignment.get(node, None)
                if id is not None:
                    for user in node.users:
                        if user in assignment and assignment.get(user, None) != id:
                            nodes_reassignment[user] = id

        for node, id in nodes_reassignment.items():
            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            assignment[node] = id
            partitions_by_id[id].add_node(node)

        # Filter single node partitions if needed
        if not self.allows_single_node_partition:
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(
                set(self.non_compute_ops or [])
            )
            partitions_to_remove = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function":
                        assert callable(node.target)
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if (
                            self.allowed_single_node_partition_ops
                            and _get_qualified_name(node.target)
                            in self.allowed_single_node_partition_ops
                        ):
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        return [p for p in partitions_by_id.values() if p.size() > 0]
