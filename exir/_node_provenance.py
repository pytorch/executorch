# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from torch.fx import Graph
from torch.fx.traceback import NodeSource, NodeSourceAction


class _NodeInfo(NamedTuple):
    name: str
    target: str
    graph_id: int


@dataclass(frozen=True, eq=False, repr=False)
class _ImmutableNodeSource(NodeSource):
    pass_name: str
    action: tuple[NodeSourceAction, ...]
    from_node: tuple["_ImmutableNodeSource", ...]
    node_info: _NodeInfo | None
    _dict: dict[str, Any] | None = field(default=None, init=False)
    _action_string: str | None = field(default=None, init=False)
    _hash: int | None = field(default=None, init=False)

    def __deepcopy__(self, memo: dict[int, Any]) -> "_ImmutableNodeSource":
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (
            NodeSource,
            (None, self.pass_name, list(self.action)),
            {
                "node_info": (
                    NodeSource.NodeInfo(*self.node_info) if self.node_info else None
                ),
                "from_node": list(self.from_node),
            },
        )

    def _get_action_string(self) -> str:
        action_string = self._action_string
        if action_string is None:
            action_string = "+".join(action.name.lower() for action in self.action)
            object.__setattr__(self, "_action_string", action_string)
        return action_string

    def _cached_dict(self) -> dict[str, Any]:
        cached = self._dict
        if cached is None:
            cached = {
                "name": self.name,
                "target": self.target,
                "graph_id": self.graph_id,
                "pass_name": self.pass_name,
                "action": self._get_action_string(),
                "from_node": [source._cached_dict() for source in self.from_node],
            }
            object.__setattr__(self, "_dict", cached)
        return cached

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._cached_dict())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ImmutableNodeSource):
            return self._cached_dict() == other._cached_dict()
        if isinstance(other, NodeSource):
            return self._cached_dict() == other.to_dict()
        return False

    def __hash__(self) -> int:
        cached_hash = self._hash
        if cached_hash is None:
            cached_hash = NodeSource.__hash__(self)
            object.__setattr__(self, "_hash", cached_hash)
        return cached_hash


def _snapshot_node_sources(
    sources: list[Any], snapshots: dict[int, NodeSource]
) -> list[Any]:
    if not any(type(source) is NodeSource for source in sources):
        return sources

    def freeze(source: Any) -> Any:
        if isinstance(source, _ImmutableNodeSource):
            return source
        if type(source) is not NodeSource:
            return source
        if id(source) not in snapshots:
            parents = tuple(freeze(parent) for parent in source.from_node)
            if any(not isinstance(parent, _ImmutableNodeSource) for parent in parents):
                snapshots[id(source)] = source
                return source
            info = source.node_info
            snapshots[id(source)] = _ImmutableNodeSource(
                source.pass_name,
                tuple(source.action),
                parents,
                _NodeInfo(info.name, info.target, info.graph_id) if info else None,
            )
        return snapshots[id(source)]

    frozen_sources = [freeze(source) for source in sources]
    if any(frozen is not source for frozen, source in zip(frozen_sources, sources)):
        return frozen_sources
    return sources


def freeze_node_provenance(graph: Graph) -> None:
    """Share immutable NodeSource ancestry without freezing unsupported metadata."""
    snapshots: dict[int, NodeSource] = {}
    for node in graph.nodes:
        if sources := node.meta.get("from_node"):
            frozen_sources = _snapshot_node_sources(sources, snapshots)
            if frozen_sources is not sources:
                node.meta["from_node"] = frozen_sources
