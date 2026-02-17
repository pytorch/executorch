# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import re
from dataclasses import dataclass
from typing import Union

from torch.fx.graph import Graph


@dataclass
class NonDelegatedNode:
    node_name: str
    num_occurrences: Union[int, None] = None


class GraphVerifier(abc.ABC):
    @abc.abstractmethod
    def verify_graph(self, graph: Graph):
        pass

    @abc.abstractmethod
    def check_num_delegated_nodes(self, num_dlg_nodes: int):
        pass


class BaseGraphVerifier(GraphVerifier):
    """Graph verifier base class. Checks for number of delegated nodes and number of selected expected nodes."""

    def __init__(
        self,
        exp_num_delegate_call_nodes: int,
        exp_non_delegated_nodes: list[NonDelegatedNode] = None,
    ):
        self.exp_non_delegated_nodes = (
            exp_non_delegated_nodes if exp_non_delegated_nodes is not None else []
        )
        self.exp_num_delegate_call_nodes = exp_num_delegate_call_nodes

    def check_num_delegated_nodes(self, num_dlg_nodes):
        assert not (
            num_dlg_nodes < self.exp_num_delegate_call_nodes
        ), f"Number of delegated nodes decreased from {self.exp_num_delegate_call_nodes} to {num_dlg_nodes}."
        assert not (
            num_dlg_nodes > self.exp_num_delegate_call_nodes
        ), f"Number of delegated nodes increased from {self.exp_num_delegate_call_nodes} to {num_dlg_nodes}."

    def verify_graph(self, graph):
        nodes = list(graph.nodes)

        # Check for specific non delegated nodes
        for exp_node in self.exp_non_delegated_nodes:
            num_exp_nodes = len(
                [node for node in nodes if exp_node.node_name in node.name]
            )
            if exp_node.num_occurrences is None:
                assert (
                    num_exp_nodes
                ), f"Graph contains no occurrences of {exp_node.node_name}."
            else:
                assert not (
                    num_exp_nodes < exp_node.num_occurrences
                ), f"Number of {exp_node.node_name} nodes decreased from {exp_node.num_occurrences} to {num_exp_nodes}."
                assert not (
                    num_exp_nodes > exp_node.num_occurrences
                ), f"Number of {exp_node.node_name} nodes increased from {exp_node.num_occurrences} to {num_exp_nodes}."

        # Check for unexpected non delegated aten nodes
        aten_fn_nodes = set(
            [
                re.split(r"_\d", node.name)[0]
                for node in nodes
                if node.name.startswith("aten")
            ]
        )
        expected_aten_fn_nodes = set(
            [exp_node.node_name for exp_node in self.exp_non_delegated_nodes]
        )
        unexpected_aten_fn_nodes = aten_fn_nodes - expected_aten_fn_nodes
        unexpected_aten_fn_nodes = "\n".join(unexpected_aten_fn_nodes)
        assert (
            not unexpected_aten_fn_nodes
        ), f"Graphs contains unexpected aten nodes:\n{unexpected_aten_fn_nodes}."
