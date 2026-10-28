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


def _reachable(nodes, next_nodes):
    """The nodes and every node reachable from them through next_nodes."""
    reached = set(nodes)
    stack = list(reached)
    while stack:
        for node in next_nodes(stack.pop()):
            if node not in reached:
                reached.add(node)
                stack.append(node)
    return reached


class _DependencyViewer:
    """Transitive users and inputs of nodes, as bitmasks over node positions.

    A whole partition can then be checked for cycles with a few integer
    operations. Nodes must be in topological order, as FX graphs keep them.
    """

    def __init__(self, graph_module: GraphModule, nodes):
        all_nodes = list(graph_module.graph.nodes)
        self.positions = {node: i for i, node in enumerate(all_nodes)}
        self.downstreams = {}
        self.upstreams = {}
        # Only these can be on a path between two of the nodes. A mask costs
        # one bit per position up to its highest node, so skipping the others
        # saves memory where most nodes are far from every partition.
        downstream_nodes = _reachable(nodes, lambda node: node.users)
        upstream_nodes = _reachable(nodes, lambda node: node.all_input_nodes)

        for node in reversed(all_nodes):
            if node in downstream_nodes:
                mask = 0
                for user in node.users:
                    mask |= (1 << self.positions[user]) | self.downstreams[user]
                self.downstreams[node] = mask

        for node in all_nodes:
            if node in upstream_nodes:
                mask = 0
                for input_node in node.all_input_nodes:
                    mask |= (1 << self.positions[input_node]) | self.upstreams[
                        input_node
                    ]
                self.upstreams[node] = mask

    def masks_of(self, nodes):
        """Bitmasks of the nodes and of all nodes downstream and upstream of them."""
        node_mask = downstreams = upstreams = 0
        for node in nodes:
            node_mask |= 1 << self.positions[node]
            downstreams |= self.downstreams[node]
            upstreams |= self.upstreams[node]
        return node_mask, downstreams, upstreams


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
        self.node_groups = (
            [set(node_group) for node_group in node_groups] if node_groups else None
        )
        self.node_to_group = collections.defaultdict(int)
        self.all_nodes_in_groups = set()
        if self.node_groups:
            for i, group in enumerate(self.node_groups):
                for node in group:
                    # Node is in multiple groups - not allowed
                    if node in self.node_to_group:
                        raise ValueError(f"Node {node} exists in multiple groups.")
                    self.node_to_group[node] = i
                    self.all_nodes_in_groups.add(node)

    def _can_merge_partitions(self, masks1, masks2):
        """Check if merging two partitions would create a cycle."""
        nodes, downstreams, upstreams = (m1 | m2 for m1, m2 in zip(masks1, masks2))
        # Fused on its own, the merged partition has a cycle exactly when a node
        # outside it is downstream of it and also upstream of it.
        return not (downstreams & upstreams & ~nodes)

    def _process_all_nodes(
        self,
        new_partition_id,
        partitions_by_id,
        assignment,
        nodes_order,
        partitions_order,
        partition_users,
        partition_map,
    ):
        """Process nodes into a partition."""
        for node in reversed(self.graph_module.graph.nodes):
            if node in assignment or not self._is_node_supported(node):
                continue

            if node in self.all_nodes_in_groups:
                group_idx = self.node_to_group[node]
                group = self.node_groups[group_idx]

                # Create a partition for group
                partition_id = next(new_partition_id)
                partition = Partition(id=partition_id, nodes=set())
                partitions_by_id[partition_id] = partition
                partitions_order[partition_id] = partition_id

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
                        target_id = assignment.get(user, None)
                        if target_id is not None and target_id != partition_id:
                            partition_map[partition_id].add(target_id)
                            partition_map[partition_id].update(partition_map[target_id])
            else:
                partition_id = next(new_partition_id)
                nodes_order[node] = partition_id
                partitions_order[partition_id] = partition_id
                partitions_by_id[partition_id] = Partition(
                    id=partition_id, nodes=[node]
                )
                assignment[node] = partition_id
                partition_users[partition_id] = set(node.users)

                # Update partition map
                for user in node.users:
                    target_id = assignment.get(user)
                    if target_id is not None:
                        partition_map[partition_id].add(target_id)
                        partition_map[partition_id].update(partition_map[target_id])

    def _merge_partitions(
        self,
        partitions_by_id,
        assignment,
        partition_users,
        partition_map,
        partitions_order,
    ):
        """Merge partitions when possible."""
        # Get current partition IDs
        partition_ids = list(partitions_by_id.keys())
        # Nothing to merge, so do not spend memory on the masks.
        if len(partition_ids) < 2:
            return

        dependency_viewer = _DependencyViewer(
            self.graph_module,
            [
                node
                for partition in partitions_by_id.values()
                for node in partition.nodes
            ],
        )
        partition_masks = {
            partition_id: dependency_viewer.masks_of(partition.nodes)
            for partition_id, partition in partitions_by_id.items()
        }

        # Set to track removed partitions from initial static list so we can skip them
        already_merged = set()
        # Try to merge each pair of partitions
        for i, p1 in enumerate(partition_ids):
            # Skip if this partition has been already merged
            if p1 in already_merged:
                continue

            for p2 in partition_ids[i + 1 :]:
                # Skip if this partition has been already merged
                if p2 in already_merged:
                    continue

                # Try to merge partitions if it doesn't create cycles
                if self._can_merge_partitions(partition_masks[p1], partition_masks[p2]):
                    self._perform_partition_merge(
                        p1,
                        p2,
                        partitions_by_id,
                        assignment,
                        partition_users,
                        partition_map,
                        partitions_order,
                    )
                    partition_masks[p1] = tuple(
                        m1 | m2
                        for m1, m2 in zip(partition_masks[p1], partition_masks.pop(p2))
                    )

                    # Mark p2 as merged
                    already_merged.add(p2)

    def _perform_partition_merge(
        self,
        p1,
        p2,
        partitions_by_id,
        assignment,
        partition_users,
        partition_map,
        partitions_order,
    ):
        """Merge partition p2 into p1."""
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
        partitions_order[p1] = min(partitions_order[p1], partitions_order[p2])

        # Remove p2
        del partitions_by_id[p2]
        del partition_users[p2]
        del partitions_order[p2]
        if p2 in partition_map:
            del partition_map[p2]

    def _process_getitem_nodes(self, partitions_by_id, assignment):
        """Post-process getitem nodes."""
        nodes_reassignment = {}

        for node in self.graph_module.graph.nodes:
            # Check if all users are getitem nodes
            is_tuple_output = True
            for user in node.users:
                if (
                    user.op != "call_function"
                    or _get_qualified_name(user.target) != "_operator.getitem"
                ):
                    is_tuple_output = False
                    break

            # Node has tuple outputs, reassign all following getitem nodes into node's partition
            if is_tuple_output:
                id = assignment.get(node, None)
                if id is not None:
                    for user in node.users:
                        if user in assignment and assignment.get(user, None) != id:
                            nodes_reassignment[user] = id

        # Apply reassignments
        for node, id in nodes_reassignment.items():
            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            assignment[node] = id
            partitions_by_id[id].add_node(node)

    def _filter_single_node_partitions(self, partitions_by_id):
        """Filter out single node partitions if needed."""
        if self.allows_single_node_partition:
            return

        default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
        non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops or []))
        partitions_to_remove = []

        for id, partition in partitions_by_id.items():
            compute_node_count = 0
            for node in partition.nodes:
                if node.op == "call_function":
                    assert callable(node.target)
                    target_name = _get_qualified_name(node.target)

                    if target_name not in non_compute_ops:
                        compute_node_count += 1

                    if (
                        self.allowed_single_node_partition_ops
                        and target_name in self.allowed_single_node_partition_ops
                    ):
                        compute_node_count += 1

            if compute_node_count <= 1:
                partitions_to_remove.append(id)

        for id in partitions_to_remove:
            del partitions_by_id[id]

    def propose_partitions(self) -> list[Partition]:
        """
        Propose partitions for the graph module based on node groups and operator support.

        Returns:
            A list of proposed partitions.
        """
        # Initialize data structures
        partition_map = collections.defaultdict(
            set
        )  # Maps partition IDs to reachable partition IDs
        assignment = {}  # Maps nodes to partition IDs
        partitions_by_id = {}  # Maps partition IDs to partitions
        nodes_order = {}  # Maps nodes to topological order
        partitions_order = {}  # Maps partition IDs to minimum topological order
        partition_users = {}  # Maps partition IDs to partition users
        new_partition_id = itertools.count()

        # Process all nodes into partitions
        self._process_all_nodes(
            new_partition_id,
            partitions_by_id,
            assignment,
            nodes_order,
            partitions_order,
            partition_users,
            partition_map,
        )

        # Merge partitions when possible
        self._merge_partitions(
            partitions_by_id,
            assignment,
            partition_users,
            partition_map,
            partitions_order,
        )

        # Post-process getitem nodes
        self._process_getitem_nodes(partitions_by_id, assignment)

        # Filter single node partitions if needed
        self._filter_single_node_partitions(partitions_by_id)

        # Return non-empty partitions
        return [p for p in partitions_by_id.values() if p.size() > 0]
