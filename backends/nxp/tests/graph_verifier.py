# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Union

from executorch.backends.nxp.neutron_partitioner import (
    NeutronPartitioner,
    NXP_DELEGATION_TAG,
)
from executorch.backends.nxp.tests.ops_aliases import (
    DequantizePerChannel,
    DequantizePerTensor,
    QuantizePerChannel,
    QuantizePerTensor,
)

from executorch.exir.dialects.edge._ops import EdgeOpOverload

from pytest_mock import MockerFixture

from torch.fx import Node
from torch.fx.graph import Graph


@dataclass
class NonDelegatedNode:
    """Represents an expected non-delegated node in the graph.

    :param node_name: The name of the node to check for
    :param num_occurrences: Expected number of occurrences. If None, just verifies that at least one exists
    """

    node_name: str
    num_occurrences: Union[int, None] = None


class GraphVerifier(abc.ABC):
    """Abstract base class for graph verification strategies."""

    @abc.abstractmethod
    def verify_graph(self, graph: Graph):
        """Verifies the graph meets expected criteria.

        :param graph: The FX graph to verify
        :raises AssertionError: If the graph does not meet expectations
        """
        pass


class BaseGraphVerifier(GraphVerifier):
    """Graph verifier base class. Checks for number of delegated nodes and number of selected expected nodes.

    This verifier performs the following checks:
    - The total number of delegated call nodes matches expectations
    - Specific non-delegated nodes appear with the expected frequency
    - No unexpected aten nodes are present in the graph
    """

    def __init__(
        self,
        exp_num_delegate_call_nodes: int,
        exp_non_delegated_nodes: list[NonDelegatedNode] = None,
    ):
        """Initializes the BaseGraphVerifier.

        :param exp_num_delegate_call_nodes: Expected number of delegated nodes
        :param exp_non_delegated_nodes: List of expected non-delegated nodes to verify
        """
        self.exp_non_delegated_nodes = (
            exp_non_delegated_nodes if exp_non_delegated_nodes is not None else []
        )
        self.exp_num_delegate_call_nodes = exp_num_delegate_call_nodes

    def check_num_delegated_nodes(self, num_dlg_nodes):
        """Checks that the number of delegated nodes matches expectations.

        :param num_dlg_nodes: Actual number of delegated nodes
        :raises AssertionError: If the count doesn't match expectations
        """
        assert not (
            num_dlg_nodes < self.exp_num_delegate_call_nodes
        ), f"Number of delegated nodes decreased from {self.exp_num_delegate_call_nodes} to {num_dlg_nodes}."
        assert not (
            num_dlg_nodes > self.exp_num_delegate_call_nodes
        ), f"Number of delegated nodes increased from {self.exp_num_delegate_call_nodes} to {num_dlg_nodes}."

    def verify_graph(self, graph):
        """Verifies the graph meets delegation and node presence expectations.

        :param graph: The FX graph to verify
        :raises AssertionError: If verification fails
        """
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


# Type alias for operators - can be either EdgeOpOverload or any callable (e.g., operator.getitem).
Operator = EdgeOpOverload | Callable


class DetailedGraphVerifier(GraphVerifier):
    """Graph verifier that checks for exact delegated and non-delegated operators.

    This verifier captures a snapshot of the graph immediately after partitioning and verifies
    that specific operators were delegated/non-delegated the expected number of times. It uses
    mocker to intercept the partition() call and create a deep copy of the nodes before they
    can be modified. Quantization/dequantization operators are ignored by default as they are
    typically not the focus of delegation verification.
    """

    default_ops_to_ignore = {
        QuantizePerTensor,
        QuantizePerChannel,
        DequantizePerTensor,
        DequantizePerChannel,
    }

    def __init__(
        self,
        mocker: MockerFixture,
        *,
        expected_delegated_ops: dict[Operator, int],
        expected_non_delegated_ops: dict[Operator, int],
        ops_to_ignore: set[Operator] | None = None,
    ):
        """Initializes the DetailedGraphVerifier and patches NeutronPartitioner.partition() to capture node state.

        :param expected_delegated_ops: Dictionary mapping operators to their expected delegation count
        :param expected_non_delegated_ops: Dictionary mapping operators to their expected non-delegation count
        :param mocker: Pytest mocker fixture for intercepting the partition method
        :param ops_to_ignore: Set of operators to ignore during verification. Defaults to quantization ops
        """
        self.expected_delegated_ops = expected_delegated_ops
        self.expected_non_delegated_ops = expected_non_delegated_ops

        self.ops_to_ignore = ops_to_ignore or self.default_ops_to_ignore

        # We need to use mocker to capture a copy of the nodes returned by NeutronPartitioner.partition() to access
        # their partition tag. The nodes in the returned graph may be modified after partition() returns, so we
        # capture a deep copy immediately when the method completes.
        self.captured_partitioned_nodes: list[Node] | None = None

        # Store original partition method for the wrapper.
        # Note: pytest-mock automatically restores the original method after the test completes,
        # so manual cleanup is not required.
        original_partition_method = NeutronPartitioner.partition

        def partition_wrapper(self_, exported_program):
            """Wraps NeutronPartitioner.partition() to capture a snapshot of nodes after partitioning.

            :param self_: The NeutronPartitioner instance
            :param exported_program: The ExportedProgram being partitioned
            :return: The PartitionResult from the original partition method
            """
            result = original_partition_method(self_, exported_program)
            # Capture a deep copy of the nodes with their metadata.
            # This ensures we have the exact state immediately after partitioning,
            # before any subsequent transformations modify the graph.
            self.captured_partitioned_nodes = list(
                deepcopy(exported_program.graph.nodes)
            )
            return result

        # Patch the partition method to intercept and capture results.
        mocker.patch.object(NeutronPartitioner, "partition", partition_wrapper)

    def verify_graph(self, graph):
        """Verifies that operators were delegated/non-delegated as expected by comparing actual counts against expectations.

        :param graph: The FX graph to verify (not directly used; we use captured nodes instead)
        :raises AssertionError: If the NeutronPartitioner wasn't used or if delegation doesn't match expectations
        """
        assert (
            self.captured_partitioned_nodes is not None
        ), "The NeutronPartitioner was not used. Cannot access delegated nodes."

        delegated_ops = defaultdict(int)
        non_delegated_ops = defaultdict(int)

        for node in self.captured_partitioned_nodes:
            # Only process call_function nodes with a target
            if not hasattr(node, "target") or node.op != "call_function":
                continue

            # Skip operators we're configured to ignore (e.g., quantization ops)
            if node.target in self.ops_to_ignore:
                continue

            # Check if the node was tagged for delegation during partitioning
            if NXP_DELEGATION_TAG in node.meta:
                delegated_ops[node.target] += 1
            else:
                non_delegated_ops[node.target] += 1

        # All ops which were either expected to be delegated, or were actually delegated.
        all_delegated_ops = list(set(self.expected_delegated_ops).union(delegated_ops))

        # All ops which were either expected to be non-delegated, or were actually non-delegated.
        all_non_delegated_ops = list(
            set(self.expected_non_delegated_ops).union(non_delegated_ops)
        )

        message = ""

        # Check delegated operators
        for op in all_delegated_ops:
            expected_count = self.expected_delegated_ops.get(op, 0)
            real_count = delegated_ops.get(op, 0)
            op_name = op.name() if hasattr(op, "name") else str(op)
            if expected_count != real_count:
                message += f"\t`{op_name}` was delegated {real_count} times instead of the expected {expected_count} times.\n"

        # Check non-delegated operators
        for op in all_non_delegated_ops:
            expected_count = self.expected_non_delegated_ops.get(op, 0)
            real_count = non_delegated_ops.get(op, 0)
            op_name = op.name() if hasattr(op, "name") else str(op)
            if expected_count != real_count:
                message += f"\t`{op_name}` was NON-delegated {real_count} times instead of the expected {expected_count} times.\n"

        if message:
            raise AssertionError(
                "Some operators were not delegated as expected:\n" + message
            )
