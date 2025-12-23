# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from typing import Any, Deque, Dict, Hashable, List, Set, Tuple, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torch.fx.node import Argument, map_arg


class FuseDuplicateUsersPass(ArmPass):
    """Fuse identical users of a producer node into a single operation.

    Example:

        y = producer(x)
        z0 = torch.add(y, bias)
        z1 = torch.add(y, bias)

    becomes a single ``torch.add`` that feeds both consumers.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        producers: Deque[Node] = deque(node for node in graph.nodes)

        while producers:
            producer = producers.popleft()

            if producer.graph is None:
                # Node was deleted by a previous rewrite while still queued.
                continue

            # Only meaningful if a value is consumed by multiple users.
            user_nodes = list(producer.users)
            if len(user_nodes) < 2:
                continue

            candidate_groups = self._get_candidate_groups(user_nodes)

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

                    user.replace_all_uses_with(representative)
                    graph.erase_node(user)
                    modified = True

                    # Revisit the current producer and the surviving user so that
                    # newly formed duplicate chains can be fused in later
                    # iterations.
                    producers.append(producer)
                    producers.append(representative)

        if modified:
            graph_module.recompile()
            graph_module.graph.lint()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)

    def _get_candidate_groups(self, user_nodes):
        users_by_target: Dict[Tuple[str, Hashable], List[Node]] = {}
        for user in user_nodes:
            if user.graph is None:
                # User might already have been removed by a prior rewrite.
                continue

            if user.op != "call_function":
                continue

            target_key = self._get_target_key(user.target)
            target_signature = (user.op, target_key)
            users_by_target.setdefault(target_signature, []).append(user)

        candidate_groups = [
            group for group in users_by_target.values() if len(group) > 1
        ]

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

        return (node.op, target_key, normalized_args, normalized_kwargs)

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
            return (
                "tensor",
                str(value.dtype),
                tuple(value.size()),
                value.device.type,
                value.requires_grad,
            )
        return value

    def _get_target_key(self, target: Any) -> Hashable:
        if isinstance(target, (EdgeOpOverload, OpOverload)):
            return str(target)
        return target
