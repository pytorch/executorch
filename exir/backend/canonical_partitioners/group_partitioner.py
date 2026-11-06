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
        all_nodes = self.nodes = list(graph_module.graph.nodes)
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

    def _compute_weight(self, node):
        """How much a node counts toward keeping its partition from being filtered."""
        if node.op != "call_function":
            return 0
        assert callable(node.target)
        target_name = _get_qualified_name(node.target)
        default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
        non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops or []))
        weight = 0 if target_name in non_compute_ops else 1
        if (
            self.allowed_single_node_partition_ops
            and target_name in self.allowed_single_node_partition_ops
        ):
            weight += 1
        return weight

    def _is_kept(self, partition, compute_weight):
        """Whether a partition will be fused into one node, if it does not grow any
        further. A single node needs no fusing: it keeps the same edges either way."""
        if self.allows_single_node_partition:
            return len(partition.nodes) > 1
        return compute_weight > 1

    def _closure(
        self, partition_id, partition_masks, assignment, nodes, kept_nodes, closures
    ):
        """Bitmask of a kept partition and of every node it reaches once each kept
        partition is fused into a single node. Cached in closures."""
        stack = [partition_id]
        while stack:
            current = stack[-1]
            if current in closures:
                stack.pop()
                continue
            current_nodes, reach, _ = partition_masks[current]
            covered = current_nodes
            missing = None
            while todo := reach & kept_nodes & ~covered:
                position = (todo & -todo).bit_length() - 1
                other = assignment[nodes[position]]
                if other not in closures:
                    missing = other
                    break
                reach |= closures[other]
                covered |= closures[other]
            if missing is None:
                closures[current] = current_nodes | reach
                stack.pop()
            elif missing in stack:
                # Caller node_groups that depend on each other: the partitions
                # on the stack reach each other, so walk each one directly.
                for p in reversed(stack):
                    closures[p] = self._walk(
                        partition_masks[p],
                        partition_masks,
                        assignment,
                        nodes,
                        kept_nodes,
                        closures,
                    )
                stack.clear()
            else:
                stack.append(missing)
        return closures[partition_id]

    def _walk(self, masks, partition_masks, assignment, nodes, kept_nodes, closures):
        """Bitmask of the given nodes and of every node they reach once each kept
        partition is fused into a single node, following the cached closures and
        the edges of the other partitions."""
        covered, reach, _ = masks
        reach |= covered
        while todo := reach & kept_nodes & ~covered:
            position = (todo & -todo).bit_length() - 1
            other = assignment[nodes[position]]
            if other in closures:
                reach |= closures[other]
                covered |= closures[other]
            else:
                other_nodes, other_reach, _ = partition_masks[other]
                reach |= other_nodes | other_reach
                covered |= other_nodes
        return reach

    def _already_on_one_cycle(
        self,
        nodes1,
        nodes2,
        partition_masks,
        assignment,
        dependency_viewer,
        kept_nodes,
        closures,
    ):
        """Whether everything in two partitions already lies on one cycle, as with
        caller node_groups that depend on each other. Merging them then puts no
        node on a cycle that was not on one before. A kept partition counts as
        one node, since it is fused, and any other partition as its own nodes."""
        nodes = dependency_viewer.nodes
        # Closure and node mask of each part that is one node in the fused graph.
        parts = []
        for n in (nodes1, nodes2):
            if n & kept_nodes:
                closure = self._closure(
                    assignment[nodes[(n & -n).bit_length() - 1]],
                    partition_masks,
                    assignment,
                    nodes,
                    kept_nodes,
                    closures,
                )
                parts.append((closure, n))
                continue
            while n:
                bit = n & -n
                n ^= bit
                closure = self._walk(
                    dependency_viewer.masks_of([nodes[bit.bit_length() - 1]]),
                    partition_masks,
                    assignment,
                    nodes,
                    kept_nodes,
                    closures,
                )
                parts.append((closure, bit))
        first_closure, first_nodes = parts[0]
        return all(
            first_closure & part_nodes and closure & first_nodes
            for closure, part_nodes in parts[1:]
        )

    def _can_merge_partitions(
        self,
        masks1,
        masks2,
        partition_masks,
        assignment,
        dependency_viewer,
        kept_nodes,
        closures,
    ):
        """Whether two partitions can be merged. No node outside the merged
        partition may be both downstream and upstream of it. Once the other kept
        partitions are fused, it must not be on a cycle through them either,
        unless everything in the two partitions already lies on one cycle."""
        nodes = dependency_viewer.nodes
        merged, downstreams, upstreams = (m1 | m2 for m1, m2 in zip(masks1, masks2))
        # Fused on its own, the merged partition has a cycle when a node outside
        # it is downstream of it and also upstream of it.
        if downstreams & upstreams & ~merged:
            return False

        # Every other kept partition becomes a single node when it is fused, so
        # reaching one of its nodes reaches everything downstream of all of
        # them. Without this, two partitions that are each free of cycles can
        # still depend on each other.
        covered = merged
        while reached := downstreams & kept_nodes & ~covered:
            position = (reached & -reached).bit_length() - 1
            closure = self._closure(
                assignment[nodes[position]],
                partition_masks,
                assignment,
                nodes,
                kept_nodes,
                closures,
            )
            downstreams |= closure
            if downstreams & upstreams & ~merged:
                return self._already_on_one_cycle(
                    masks1[0],
                    masks2[0],
                    partition_masks,
                    assignment,
                    dependency_viewer,
                    kept_nodes,
                    closures,
                )
            covered |= closure
        return True

    def _process_all_nodes(
        self,
        new_partition_id,
        partitions_by_id,
        assignment,
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

                # Add all supported nodes from the group to the partition
                for node in group:
                    if self._is_node_supported(node):
                        partition.add_node(node)
                        assignment[node] = partition_id
            else:
                partition_id = next(new_partition_id)
                partitions_by_id[partition_id] = Partition(
                    id=partition_id, nodes=[node]
                )
                assignment[node] = partition_id

    def _merge_partitions(self, partitions_by_id, assignment):
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
        compute_weights = {
            partition_id: sum(self._compute_weight(n) for n in partition.nodes)
            for partition_id, partition in partitions_by_id.items()
        }
        # Nodes of the partitions that will be fused, so a merge check can
        # follow paths through them.
        kept_nodes = 0
        for partition_id, partition in partitions_by_id.items():
            if self._is_kept(partition, compute_weights[partition_id]):
                kept_nodes |= partition_masks[partition_id][0]
        # Closures of kept partitions, see _closure.
        closures = {}

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

                if self._can_merge_partitions(
                    partition_masks[p1],
                    partition_masks[p2],
                    partition_masks,
                    assignment,
                    dependency_viewer,
                    kept_nodes,
                    closures,
                ):
                    self._perform_partition_merge(p1, p2, partitions_by_id, assignment)
                    partition_masks[p1] = tuple(
                        m1 | m2
                        for m1, m2 in zip(partition_masks[p1], partition_masks.pop(p2))
                    )
                    compute_weights[p1] += compute_weights.pop(p2)
                    merged_nodes = partition_masks[p1][0]
                    if self._is_kept(partitions_by_id[p1], compute_weights[p1]):
                        kept_nodes |= merged_nodes
                    # Only closures that reach the merged nodes change.
                    stale = [p for p, c in closures.items() if c & merged_nodes]
                    for p in stale:
                        del closures[p]

                    # Mark p2 as merged
                    already_merged.add(p2)

    def _perform_partition_merge(self, p1, p2, partitions_by_id, assignment):
        """Merge partition p2 into p1."""
        # Merge p2 into p1
        partitions_by_id[p1].nodes.update(partitions_by_id[p2].nodes)
        for node in partitions_by_id[p2].nodes:
            assignment[node] = p1

        # Remove p2
        del partitions_by_id[p2]

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

        partitions_to_remove = []

        for id, partition in partitions_by_id.items():
            compute_node_count = sum(self._compute_weight(n) for n in partition.nodes)
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
        assignment = {}  # Maps nodes to partition IDs
        partitions_by_id = {}  # Maps partition IDs to partitions
        new_partition_id = itertools.count()

        # Process all nodes into partitions
        self._process_all_nodes(new_partition_id, partitions_by_id, assignment)

        # Merge partitions when possible
        self._merge_partitions(partitions_by_id, assignment)

        # Post-process getitem nodes
        self._process_getitem_nodes(partitions_by_id, assignment)

        # Filter single node partitions if needed
        self._filter_single_node_partitions(partitions_by_id)

        # Return non-empty partitions
        return [p for p in partitions_by_id.values() if p.size() > 0]
