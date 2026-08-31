# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections import deque
from collections.abc import Callable
from typing import Any, Deque, Dict, Hashable, List, Set, Tuple, Type

import torch
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torch.fx.node import Argument, map_arg


class FuseDuplicateUsersPass(ExportPass):
    """Fuse identical users of a producer node into a single operation.

    Example:

        y = producer(x)
        z0 = torch.add(y, bias)
        z1 = torch.add(y, bias)

    becomes a single ``torch.add`` that feeds both consumers.

    Args:
        excluded_targets: Operator targets that must never be fused.
        may_alias_outputs: Whether independently returned duplicate values may
            be merged. Callers enabling this must restore unique output nodes
            before serialization.
        semantic_key: Optional backend metadata included in the duplicate
            signature, represented as an FX argument tree.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    _recompile_before_retrace = True

    def __init__(
        self,
        excluded_targets: frozenset | None = None,
        may_alias_outputs: bool = False,
        *,
        semantic_key: Callable[[Node], Argument] | None = None,
    ) -> None:
        super().__init__()
        # Fusing duplicate users of an op collapses them onto one node. That is
        # unsound for any op whose consumers a later stage assumes are distinct,
        # so backends name those here rather than the pass guessing.
        self._excluded_targets = excluded_targets or frozenset()
        self._semantic_key = semantic_key
        self._may_alias_outputs = may_alias_outputs

    def call(self, graph_module: GraphModule) -> PassResult:  # noqa: C901
        graph = graph_module.graph
        modified = False

        graph_nodes = list(graph.nodes)
        node_order = {node: index for index, node in enumerate(graph_nodes)}
        active_nodes = set(graph_nodes)
        effect_prefix = [0]
        for node in graph_nodes:
            effect_prefix.append(
                effect_prefix[-1] + int(self._has_observable_effect(node))
            )
        returned = self._returned_nodes(graph)
        downstream_mutation: Dict[Node, bool] = {}
        producers: Deque[Node] = deque(graph_nodes)

        while producers:
            producer = producers.popleft()

            if producer not in active_nodes:
                # Node was deleted by a previous rewrite while still queued.
                continue

            # Only meaningful if a value is consumed by multiple users.
            user_nodes = list(producer.users)
            if len(user_nodes) < 2:
                continue

            candidate_groups = self._get_candidate_groups(
                node_order, active_nodes, user_nodes
            )

            signature_to_user: Dict[Tuple[Hashable, ...], Node] = {}
            for group in candidate_groups:
                for user in group:
                    signature = self._build_user_signature(user)
                    if signature is None:
                        continue

                    representative = signature_to_user.get(signature)
                    if representative is None:
                        # Check if we already encountered identical node that we can fuse with.
                        signature_to_user[signature] = user
                        continue

                    if user is representative:
                        # The queue can enqueue the surviving node again after rewrites.
                        continue

                    if self._has_intervening_effect(
                        representative, user, node_order, effect_prefix
                    ):
                        signature_to_user[signature] = user
                        continue
                    if not self._can_share_result(
                        representative, user, downstream_mutation, returned
                    ):
                        continue

                    user.replace_all_uses_with(representative)
                    graph.erase_node(user)
                    active_nodes.remove(user)
                    if user in returned:
                        returned.remove(user)
                        returned.add(representative)
                    modified = True

                    # Revisit the current producer and the surviving user so that
                    # newly formed duplicate chains can be fused in later
                    # iterations.
                    producers.append(producer)
                    producers.append(representative)

        if modified:
            if self._recompile_before_retrace:
                graph_module.recompile()
            graph_module.graph.lint()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)

    def _can_share_result(
        self,
        representative: Node,
        user: Node,
        downstream_mutation: Dict[Node, bool],
        returned: Set[Node],
    ) -> bool:
        for node in (representative, user):
            if node not in downstream_mutation:
                downstream_mutation[node] = self._has_downstream_mutation(node)
            if downstream_mutation[node]:
                return False

        user_reaches_output = self._reaches_output_through_aliases(user, returned)
        representative_reaches_output = self._reaches_output_through_aliases(
            representative, returned
        )
        if not user_reaches_output or not representative_reaches_output:
            return True
        return (
            self._may_alias_outputs
            and user in returned
            and representative in returned
            and not self._reaches_output_through_aliases(user, returned - {user})
            and not self._reaches_output_through_aliases(
                representative, returned - {representative}
            )
        )

    def _get_candidate_groups(self, node_order, active_nodes, user_nodes):
        users_by_target: Dict[Tuple[str, Hashable], List[Node]] = {}
        for user in user_nodes:
            if user not in active_nodes:
                # User might already have been removed by a prior rewrite.
                continue

            if user.op != "call_function":
                continue

            if user.target in self._excluded_targets:
                continue

            if not self._is_safe_to_fuse(user):
                continue

            target_key = self._get_target_key(user.target)
            target_signature = (user.op, target_key)
            users_by_target.setdefault(target_signature, []).append(user)

        candidate_groups = []
        for group in users_by_target.values():
            if len(group) > 1:
                candidate_groups.append(
                    sorted(group, key=lambda node: node_order[node])
                )

        return candidate_groups

    def _build_user_signature(self, node: Node) -> Tuple[Hashable, ...] | None:
        try:
            normalized_args = self._to_hashable(
                map_arg(node.args, self._map_leaf_to_key)
            )
            normalized_kwargs = self._to_hashable(
                {k: map_arg(v, self._map_leaf_to_key) for k, v in node.kwargs.items()}
            )
        except TypeError:
            return None

        target_key = self._get_target_key(node.target)
        semantic_key = None
        if self._semantic_key is not None:
            try:
                semantic_key = self._to_hashable(self._semantic_key(node))
            except TypeError:
                return None

        return (
            node.op,
            target_key,
            normalized_args,
            normalized_kwargs,
            semantic_key,
        )

    @staticmethod
    def _is_safe_to_fuse(node: Node) -> bool:
        target = node.target
        tags = {
            *getattr(target, "tags", ()),
            *getattr(getattr(target, "_op", None), "tags", ()),
        }
        if any(
            getattr(tag, "name", "").startswith("nondeterministic")
            or getattr(tag, "name", "") == "inplace"
            for tag in tags
        ):
            return False

        schema = FuseDuplicateUsersPass._schema(target)
        if schema is None:
            return target is operator.getitem
        if schema.is_mutable:
            return False
        return not any(result.alias_info is not None for result in schema.returns)

    @staticmethod
    def _schema(target: Any) -> Any | None:
        return getattr(target, "_schema", None) or getattr(
            getattr(target, "_op", None), "_schema", None
        )

    @classmethod
    def _has_intervening_effect(
        cls,
        first: Node,
        second: Node,
        node_order: Dict[Node, int],
        effect_prefix: List[int],
    ) -> bool:
        start, end = sorted((node_order[first], node_order[second]))
        return effect_prefix[end] != effect_prefix[start + 1]

    @classmethod
    def _has_observable_effect(cls, node: Node) -> bool:
        if node.op not in {"call_function", "call_method", "call_module"}:
            return False
        if node.op != "call_function":
            return True

        target = node.target
        if target is operator.getitem:
            return False
        tags = {
            *getattr(target, "tags", ()),
            *getattr(getattr(target, "_op", None), "tags", ()),
        }
        if any(
            getattr(tag, "name", "").startswith("nondeterministic")
            or getattr(tag, "name", "") == "inplace"
            for tag in tags
        ):
            return True

        schema = cls._schema(target)
        return schema is None or schema.is_mutable

    @classmethod
    def _has_downstream_mutation(cls, node: Node) -> bool:
        pending = [node]
        visited: Set[Node] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for user in current.users:
                if cls._may_mutate_input(user, current):
                    return True
                if cls._may_alias_input(user, current):
                    pending.append(user)
        return False

    @classmethod
    def _may_mutate_input(cls, user: Node, input_node: Node) -> bool:
        if input_node not in user.all_input_nodes:
            return False
        if user.op in {"call_method", "call_module"}:
            return True
        if user.op != "call_function" or user.target is operator.getitem:
            return False
        schema = cls._schema(user.target)
        return schema is None or schema.is_mutable

    @classmethod
    def _reaches_output_through_aliases(cls, node: Node, returned: Set[Node]) -> bool:
        pending = [node]
        visited: Set[Node] = set()
        while pending:
            current = pending.pop()
            if current in returned:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(
                user for user in current.users if cls._may_alias_input(user, current)
            )
        return False

    @classmethod
    def _may_alias_input(cls, user: Node, input_node: Node) -> bool:
        if user.op != "call_function" or input_node not in user.all_input_nodes:
            return False
        if user.target is operator.getitem:
            return True
        schema = cls._schema(user.target)
        return schema is not None and any(
            result.alias_info is not None for result in schema.returns
        )

    @staticmethod
    def _returned_nodes(graph: torch.fx.Graph) -> Set[Node]:
        output_node = graph.output_node()
        returned: Set[Node] = set()
        map_arg((output_node.args, output_node.kwargs), lambda node: returned.add(node))
        return returned

    def _map_leaf_to_key(self, node: Node) -> Argument:
        return node.name

    def _to_hashable(self, value: Any) -> Hashable:
        """Convert arbitrarily nested structures into hashable tuples."""

        if isinstance(value, (list, tuple)):
            return tuple(self._to_hashable(v) for v in value)
        if isinstance(value, dict):
            normalized_items = [(k, self._to_hashable(v)) for k, v in value.items()]
            return tuple(sorted(normalized_items, key=lambda item: repr(item[0])))
        if isinstance(value, set):
            hashable_values: List[Hashable] = [self._to_hashable(v) for v in value]
            return tuple(sorted(hashable_values, key=repr))
        if isinstance(value, slice):
            return (
                "slice",
                self._to_hashable(value.start),
                self._to_hashable(value.stop),
                self._to_hashable(value.step),
            )
        if isinstance(value, range):
            return ("range", value.start, value.stop, value.step)
        if isinstance(value, torch.Size):
            return ("size", tuple(value))
        if isinstance(value, torch.dtype):
            return ("dtype", str(value))
        if isinstance(value, torch.device):
            return ("device", str(value))
        if isinstance(value, torch.memory_format):
            return ("memory_format", str(value))
        if isinstance(value, torch.Tensor):
            # Distinct literal tensors can have identical metadata but different
            # values, so only the same tensor object represents the same argument.
            return ("tensor", id(value))
        return value

    def _get_target_key(self, target: Any) -> Hashable:
        if isinstance(target, (EdgeOpOverload, OpOverload)):
            return str(target)
        return target
