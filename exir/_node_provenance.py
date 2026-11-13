# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
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

    def __deepcopy__(self, memo: dict[int, Any]) -> "_ImmutableNodeSource":
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return type(self), (self.pass_name, self.action, self.from_node, self.node_info)

    def _get_action_string(self) -> str:
        return "+".join(action.name.lower() for action in self.action)

    def to_dict(self) -> dict[str, Any]:
        # NodeSource caches a mutable dict; snapshots must not expose shared state.
        return {
            "name": self.name,
            "target": self.target,
            "graph_id": self.graph_id,
            "pass_name": self.pass_name,
            "action": self._get_action_string(),
            "from_node": [source.to_dict() for source in self.from_node],
        }


def freeze_node_provenance(graph: Graph) -> None:
    """Snapshot provenance once, sharing immutable ancestry across pass replays.

    FX deep-copies inherited NodeSource records when it creates new provenance.
    Immutable snapshots retain that history without recursively copying it again.
    The outer lists remain independent and mutable.
    """
    snapshots: dict[int, _ImmutableNodeSource] = {}

    def freeze(source: NodeSource) -> _ImmutableNodeSource:
        if isinstance(source, _ImmutableNodeSource):
            return source
        if id(source) not in snapshots:
            info = source.node_info
            snapshots[id(source)] = _ImmutableNodeSource(
                source.pass_name,
                tuple(source.action),
                tuple(freeze(parent) for parent in source.from_node),
                _NodeInfo(info.name, info.target, info.graph_id) if info else None,
            )
        return snapshots[id(source)]

    for node in graph.nodes:
        if sources := node.meta.get("from_node"):
            node.meta["from_node"] = [freeze(source) for source in sources]
