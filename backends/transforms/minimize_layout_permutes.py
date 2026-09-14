# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections import deque
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.fx
from executorch.backends.transforms.channels_last_layout import (
    is_permute_copy,
    PERMUTE_COPY_TARGETS,
)
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


@dataclass
class _FlowEdge:
    destination: int
    reverse: int
    capacity: int


class _FlowNetwork:
    def __init__(self, size: int) -> None:
        self._edges: list[list[_FlowEdge]] = [[] for _ in range(size)]

    def add_edge(self, source: int, destination: int, capacity: int) -> None:
        forward = _FlowEdge(destination, len(self._edges[destination]), capacity)
        reverse = _FlowEdge(source, len(self._edges[source]), 0)
        self._edges[source].append(forward)
        self._edges[destination].append(reverse)

    def add_cut_edge(self, lhs: int, rhs: int, capacity: int) -> None:
        self.add_edge(lhs, rhs, capacity)
        self.add_edge(rhs, lhs, capacity)

    def minimum_cut(self, source: int, sink: int) -> tuple[int, set[int]]:
        flow = 0
        while True:
            parents: list[tuple[int, int] | None] = [None] * len(self._edges)
            queue = deque([source])
            while queue and parents[sink] is None:
                node = queue.popleft()
                for edge_index, edge in enumerate(self._edges[node]):
                    if (
                        edge.capacity > 0
                        and edge.destination != source
                        and parents[edge.destination] is None
                    ):
                        parents[edge.destination] = (node, edge_index)
                        queue.append(edge.destination)
            if parents[sink] is None:
                break

            path_capacity = 1 << 60
            node = sink
            while node != source:
                parent, edge_index = cast(tuple[int, int], parents[node])
                path_capacity = min(
                    path_capacity, self._edges[parent][edge_index].capacity
                )
                node = parent
            node = sink
            while node != source:
                parent, edge_index = cast(tuple[int, int], parents[node])
                edge = self._edges[parent][edge_index]
                edge.capacity -= path_capacity
                self._edges[node][edge.reverse].capacity += path_capacity
                node = parent
            flow += path_capacity

        reachable = {source}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for edge in self._edges[node]:
                if edge.capacity > 0 and edge.destination not in reachable:
                    reachable.add(edge.destination)
                    queue.append(edge.destination)
        return flow, reachable


@dataclass
class _MinCutPlan:
    subgraph: RemovePermutesAroundElementwiseOps.Subgraph
    edges_in_to_permute: set[tuple[torch.fx.Node, torch.fx.Node]] = field(
        default_factory=set
    )
    edges_out_to_permute: set[tuple[torch.fx.Node, torch.fx.Node]] = field(
        default_factory=set
    )


# A slot is the path of keys from a node's args/kwargs down to one operand
# occurrence, e.g. ("args", 0, 2) for the third tensor of a cat list.
_Slot = tuple[str | int, ...]
_BoundarySlots = dict[torch.fx.Node, dict[torch.fx.Node, list[_Slot]]]


class MinimizeLayoutPermutes(RemovePermutesAroundElementwiseOps):
    """Select globally profitable layouts before running the legacy cleanup."""

    def _is_min_cut_candidate(self, node: torch.fx.Node, rank: int) -> bool:
        if self._is_squeeze_unsqueeze_view(node) or self._interleave_triple(node):
            return False
        if not self.is_node_permutable(node):
            return False
        shape = self._concrete_shape(node)
        return shape is not None and len(shape) == rank

    def _min_cut_component(self, seed: torch.fx.Node, rank: int) -> set[torch.fx.Node]:
        component: set[torch.fx.Node] = set()
        pending = [seed]
        while pending:
            node = pending.pop()
            if node in component or not self._is_min_cut_candidate(node, rank):
                continue
            component.add(node)
            for adjacent in (*node.all_input_nodes, *node.users):
                if adjacent not in component and self._is_min_cut_candidate(
                    adjacent, rank
                ):
                    pending.append(adjacent)
        return component

    @staticmethod
    def _add_or_original_cost(
        network: _FlowNetwork,
        source: int,
        node_ids: dict[torch.fx.Node, int],
        auxiliary: int,
        nodes: set[torch.fx.Node],
        infinite: int,
    ) -> None:
        network.add_edge(source, auxiliary, 1)
        for node in nodes:
            network.add_edge(auxiliary, node_ids[node], infinite)

    @staticmethod
    def _add_or_permuted_cost(
        network: _FlowNetwork,
        sink: int,
        node_ids: dict[torch.fx.Node, int],
        auxiliary: int,
        nodes: set[torch.fx.Node],
        infinite: int,
    ) -> None:
        network.add_edge(auxiliary, sink, 1)
        for node in nodes:
            network.add_edge(node_ids[node], auxiliary, infinite)

    def _minimum_cut_selection(  # noqa: C901
        self,
        component: set[torch.fx.Node],
        start_permute: list[int],
    ) -> tuple[set[torch.fx.Node], int] | None:
        end_permute = [start_permute.index(i) for i in range(len(start_permute))]
        original_cost = {node: 0 for node in component}
        permuted_cost = {node: 0 for node in component}
        pairwise_edges: set[tuple[torch.fx.Node, torch.fx.Node]] = set()
        incoming_permute_groups: dict[torch.fx.Node, set[torch.fx.Node]] = {}
        incoming_direct_groups: dict[torch.fx.Node, set[torch.fx.Node]] = {}
        baseline = 0

        for node in component:
            for inp in node.all_input_nodes:
                if inp in component:
                    pairwise_edges.add((inp, node))
                    continue
                inp_val = inp.meta.get("val")
                if inp_val is None:
                    return None
                if inp.target in PERMUTE_COPY_TARGETS:
                    if self.get_permutation(inp) == start_permute:
                        incoming_permute_groups.setdefault(inp, set()).add(node)
                    else:
                        permuted_cost[node] += 1 << 30
                    continue
                if inp_val.numel() == 1:
                    continue
                if self._is_permutation_sink_view(inp):
                    if (
                        len(inp.users) == 1
                        and self._remapped_sink_shape(inp, start_permute) is not None
                    ):
                        continue
                    permuted_cost[node] += 1 << 30
                    continue
                if self._is_constant(inp):
                    rank = self._get_node_rank(inp)
                    if rank is None or rank > len(start_permute):
                        permuted_cost[node] += 1 << 30
                    elif rank == len(start_permute):
                        incoming_direct_groups.setdefault(inp, set()).add(node)
                    continue
                if len(inp_val.shape) != len(start_permute):
                    permuted_cost[node] += 1 << 30
                    continue
                incoming_direct_groups.setdefault(inp, set()).add(node)

            has_direct_user = False
            for user in node.users:
                if user in component:
                    continue
                if user.target in PERMUTE_COPY_TARGETS:
                    if self.get_permutation(user) == end_permute:
                        original_cost[node] += 1
                        baseline += 1
                    continue
                has_direct_user = True
            if has_direct_user:
                permuted_cost[node] += 1

        original_groups: list[set[torch.fx.Node]] = []
        for permute, consumers in incoming_permute_groups.items():
            if set(permute.users).issubset(component):
                baseline += 1
                original_groups.append(consumers)

        permuted_groups = list(incoming_direct_groups.values())
        ordered_nodes = sorted(component)
        node_ids = {node: index for index, node in enumerate(ordered_nodes)}
        auxiliary_count = len(original_groups) + len(permuted_groups)
        source = len(ordered_nodes) + auxiliary_count
        sink = source + 1
        infinite = 1 << 30
        network = _FlowNetwork(sink + 1)

        for node, node_id in node_ids.items():
            network.add_edge(source, node_id, original_cost[node])
            network.add_edge(node_id, sink, permuted_cost[node])
        for producer, consumer in pairwise_edges:
            network.add_cut_edge(node_ids[producer], node_ids[consumer], 1)

        auxiliary = len(ordered_nodes)
        for nodes in original_groups:
            self._add_or_original_cost(
                network, source, node_ids, auxiliary, nodes, infinite
            )
            auxiliary += 1
        for nodes in permuted_groups:
            self._add_or_permuted_cost(
                network, sink, node_ids, auxiliary, nodes, infinite
            )
            auxiliary += 1

        cost, reachable = network.minimum_cut(source, sink)
        if cost >= baseline:
            return None
        selected = {node for node, node_id in node_ids.items() if node_id in reachable}
        if not selected:
            return None
        return selected, baseline - cost

    def _build_min_cut_plan(
        self,
        selected: set[torch.fx.Node],
        start_permute: list[int],
    ) -> _MinCutPlan | None:
        end_permute = [start_permute.index(i) for i in range(len(start_permute))]
        subgraph = self.Subgraph(start_permute, end_permute, nodes=selected)
        plan = _MinCutPlan(subgraph)
        for node in selected:
            subgraph.node_end_permute[node] = end_permute
            subgraph.node_start_permute[node] = start_permute
            if not self._add_plan_inputs(plan, node, selected):
                return None
            if not self._add_plan_outputs(plan, node, selected):
                return None
        return plan

    def _add_plan_inputs(
        self,
        plan: _MinCutPlan,
        node: torch.fx.Node,
        selected: set[torch.fx.Node],
    ) -> bool:
        subgraph = plan.subgraph
        for inp in node.all_input_nodes:
            if inp in selected:
                continue
            inp_val = inp.meta.get("val")
            if inp_val is None:
                return False
            if inp.target in PERMUTE_COPY_TARGETS:
                if self.get_permutation(inp) != subgraph.start_permute:
                    return False
                subgraph.edges_in.add((inp, node))
                continue
            if inp_val.numel() == 1:
                continue
            if self._is_permutation_sink_view(inp):
                remapped_shape = self._remapped_sink_shape(inp, subgraph.start_permute)
                if remapped_shape is None or len(inp.users) != 1:
                    return False
                subgraph.view_shape_overrides[inp] = remapped_shape
                subgraph.sink_edges_in.add((inp, node))
            elif self._is_constant(inp) and len(inp_val.shape) < len(
                subgraph.start_permute
            ):
                subgraph.constant_edges_in.add((inp, node))
            elif len(inp_val.shape) != len(subgraph.start_permute):
                # A boundary permute is only expressible at the region's rank, so
                # a broadcast operand of a different rank cannot be compensated.
                return False
            else:
                plan.edges_in_to_permute.add((inp, node))
        return True

    def _add_plan_outputs(
        self,
        plan: _MinCutPlan,
        node: torch.fx.Node,
        selected: set[torch.fx.Node],
    ) -> bool:
        subgraph = plan.subgraph
        for user in node.users:
            if user in selected:
                continue
            if user.target not in PERMUTE_COPY_TARGETS:
                plan.edges_out_to_permute.add((node, user))
                continue
            user_permute = self.get_permutation(user)
            if user_permute == subgraph.end_permute:
                subgraph.edges_out.add((node, user))
            elif user_permute is not None:
                subgraph.edges_out_to_update.add(
                    (
                        node,
                        user,
                        tuple(subgraph.start_permute[dim] for dim in user_permute),
                    )
                )
            else:
                return False
        return True

    def _find_min_cut_plans(
        self, graph_module: torch.fx.GraphModule
    ) -> list[_MinCutPlan]:
        candidates: list[tuple[int, _MinCutPlan]] = []
        seen: set[tuple[frozenset[torch.fx.Node], tuple[int, ...]]] = set()
        for permute in graph_module.graph.nodes:
            if not is_permute_copy(permute):
                continue
            start_permute = self.get_permutation(permute)
            if start_permute is None:
                continue
            for user in permute.users:
                if not self._is_min_cut_candidate(user, len(start_permute)):
                    continue
                component = self._min_cut_component(user, len(start_permute))
                key = (frozenset(component), tuple(start_permute))
                if key in seen:
                    continue
                seen.add(key)
                selection = self._minimum_cut_selection(component, start_permute)
                if selection is None:
                    continue
                selected, savings = selection
                plan = self._build_min_cut_plan(selected, start_permute)
                if plan is not None:
                    candidates.append((savings, plan))

        claimed: set[torch.fx.Node] = set()
        plans: list[_MinCutPlan] = []
        for _, plan in sorted(candidates, key=lambda item: item[0], reverse=True):
            if claimed.isdisjoint(plan.subgraph.nodes):
                plans.append(plan)
                claimed.update(plan.subgraph.nodes)
        return plans

    @staticmethod
    def _direct_edges_are_current(plan: _MinCutPlan) -> bool:
        for inp, user in (
            *plan.edges_in_to_permute,
            *plan.edges_out_to_permute,
        ):
            if inp not in user.all_input_nodes or inp.meta.get("val") is None:
                return False
        return True

    def _apply_min_cut_plan(self, plan: _MinCutPlan) -> bool:
        if not self._direct_edges_are_current(plan):
            return False
        # Resolve the operand slots before the region is rewritten. Bypassing an
        # incoming permute can alias a boundary operand onto a node the same user
        # already consumes, after which the two are indistinguishable by value.
        slots_in = self._boundary_slots(plan.edges_in_to_permute)
        slots_out = self._boundary_slots(plan.edges_out_to_permute)
        if not self.permute_subgraph(plan.subgraph):
            return False
        self._insert_boundary_permutes(
            slots_in,
            plan.subgraph.end_permute,
            output_uses_original_meta=False,
        )
        self._insert_boundary_permutes(
            slots_out,
            plan.subgraph.start_permute,
            output_uses_original_meta=True,
        )
        return True

    @classmethod
    def _boundary_slots(
        cls, edges: set[tuple[torch.fx.Node, torch.fx.Node]]
    ) -> _BoundarySlots:
        slots: _BoundarySlots = {}
        for inp, user in edges:
            paths = cls._operand_slots(user, inp)
            if paths:
                slots.setdefault(inp, {})[user] = paths
        return slots

    @staticmethod
    def _operand_slots(user: torch.fx.Node, inp: torch.fx.Node) -> list[_Slot]:
        paths: list[_Slot] = []

        def walk(value, path: _Slot) -> None:
            if value is inp:
                paths.append(path)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    walk(item, (*path, index))
            elif isinstance(value, dict):
                for key, item in value.items():
                    walk(item, (*path, key))

        walk(user.args, ("args",))
        walk(user.kwargs, ("kwargs",))
        return paths

    @staticmethod
    def _replace_operand_slots(
        user: torch.fx.Node, paths: list[_Slot], replacement: torch.fx.Node
    ) -> None:
        def substitute(value, path: _Slot):
            if not path:
                return replacement
            key, rest = path[0], path[1:]
            if isinstance(value, dict):
                updated = dict(value)
                updated[key] = substitute(value[key], rest)
                return updated
            items = list(value)
            items[key] = substitute(items[key], rest)
            return tuple(items) if isinstance(value, tuple) else items

        args, kwargs = user.args, user.kwargs
        for path in paths:
            if path[0] == "args":
                args = substitute(args, path[1:])
            else:
                kwargs = substitute(kwargs, path[1:])
        user.args = args
        user.kwargs = kwargs

    @classmethod
    def _insert_boundary_permutes(
        cls,
        slots: _BoundarySlots,
        permutation: list[int],
        output_uses_original_meta: bool,
    ) -> None:
        for inp, users in slots.items():
            inp_val = inp.meta.get("val")
            if inp_val is None:
                continue
            with inp.graph.inserting_before(min(users)):
                new_node = inp.graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.permute_copy.default,
                    args=(inp, permutation),
                )
            new_node.meta["val"] = (
                inp_val if output_uses_original_meta else inp_val.permute(permutation)
            )
            for user, paths in users.items():
                cls._replace_operand_slots(user, paths, new_node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._sq_unsq_cache.clear()
        self._interleave_cache.clear()
        modified = False
        for plan in self._find_min_cut_plans(graph_module):
            modified |= self._apply_min_cut_plan(plan)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = ExportPass.call(self, graph_module).graph_module

        legacy_result = super().call(graph_module)
        return PassResult(
            legacy_result.graph_module, modified or legacy_result.modified
        )
