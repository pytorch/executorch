# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict

import pandas as pd
import torch


# Column names of the DataFrame returned by DelegationInfo.get_operator_delegation_dataframe()
# which describes the summarized delegation information grouped by each operator type
_OCCURRENCES_IN_DELEGATED_GRAPHS = "occurrences_in_delegated_graphs"
_OCCURRENCES_IN_NON_DELEGATED_GRAPHS = "occurrences_in_non_delegated_graphs"


@dataclass
class DelegationBreakdown:
    """
    DelegationBreakdown contains the number of delegated and non-delegated nodes
    of the operator type op_type.

    Args:
        delegated: The number of delegated nodes.
        non_delegated: The number of non-delegated nodes.
    """

    op_type: str = ""
    delegated: int = 0
    non_delegated: int = 0


@dataclass
class DelegationInfo:
    """
    DelegationInfo contains information of a delegated graph module.

    Args:
        num_delegated_subgraphs: The number of delegated subgraphs.
        num_delegated_nodes: The number of delegated nodes.
        num_non_delegated_nodes: The number of non-delegated nodes.
        delegation_by_operator: A dictionary of operator type to DelegationBreakdown.
    """

    num_delegated_subgraphs: int
    num_delegated_nodes: int
    num_non_delegated_nodes: int
    delegation_by_operator: Dict[str, DelegationBreakdown]

    def get_summary(self) -> str:
        """
        Get a summary of the delegation information in string format.

        Args:
            None

        Returns:
            A string containing information of some class attributes for easy print-out.
        """

        # Assemble and return the summary string
        summary_str = f"Total delegated subgraphs: {self.num_delegated_subgraphs}\n"
        summary_str += f"Number of delegated nodes: {self.num_delegated_nodes}\n"
        summary_str += (
            f"Number of non-delegated nodes: {self.num_non_delegated_nodes}\n"
        )
        return summary_str

    def get_operator_delegation_dataframe(self) -> pd.DataFrame:
        """
        Get the delegation information grouped by operator type in a pandas DataFrame.

        Args:
            None

        Returns:
            Returns a pandas DataFrame containing the following columns:
            - op_type: The operator type, with the last row being "Total".
            - occurrences_in_delegated_graphs: The number of occurrences of the op_type in delegated subgraphs.
            - occurrences_in_non_delegated_graphs: The number of occurrences of the op_type not in delegated subgraphs.
            With the last row being the total number of delegated and non-delegated occurrences of each op_type.
        """

        # Convert the dict to a dataframe
        list_of_dicts = [
            asdict(breakdown) for breakdown in self.delegation_by_operator.values()
        ]
        df = pd.DataFrame(list_of_dicts)
        # Rename columns for better understandability
        df = df.rename(
            columns={
                "delegated": _OCCURRENCES_IN_DELEGATED_GRAPHS,
                "non_delegated": _OCCURRENCES_IN_NON_DELEGATED_GRAPHS,
            }
        )
        df = df.sort_values(by="op_type", ignore_index=True)

        # Add a Total row at the bottom
        total_delegated_nodes = df[_OCCURRENCES_IN_DELEGATED_GRAPHS].sum()
        total_non_delegated_nodes = df[_OCCURRENCES_IN_NON_DELEGATED_GRAPHS].sum()
        df.loc[len(df)] = ["Total", total_delegated_nodes, total_non_delegated_nodes]

        return df


def get_delegation_info(
    graph_module: torch.fx.GraphModule,
) -> DelegationInfo:
    """
    Util function to get the delegation information of the given graph module.

    Args:
        graph_module: The lowered graph module to get the delegation information from.

    Returns:
        Return a DelegationInfo object containing the delegation information.
    """

    def _get_op_type(node_name: str) -> str:
        # node_name is in format <op_type> or <op_type>_x in which x is an integer suffix.
        return re.sub(r"_[\d]+$", "", node_name)

    op_occurrences_dict = defaultdict(lambda: DelegationBreakdown())

    def _insert_op_occurrences_dict(node_name: str, delegated: bool) -> None:
        op_type = _get_op_type(node_name)
        op_occurrences_dict[op_type].op_type = op_type
        if delegated:
            op_occurrences_dict[op_type].delegated += 1
        else:
            op_occurrences_dict[op_type].non_delegated += 1

    delegated_subgraph_counter = 0

    lowered_module_dict = {
        node.name: getattr(graph_module, node.name)
        for node in graph_module.graph.nodes
        if node.op == "get_attr" and node.name.startswith("lowered_module_")
    }

    for node in graph_module.graph.nodes:
        if (
            node.op == "call_function"
            and _get_op_type(node.name) != "executorch_call_delegate"
        ):
            # Non-delegated node
            _insert_op_occurrences_dict(node_name=node.name, delegated=False)
        # Check if the node is a lowered module
        if node.op == "get_attr" and node.name.startswith("lowered_module_"):
            lowered_module = lowered_module_dict[node.name]
            delegated_subgraph_counter += 1
            for node_in_lowered_module in lowered_module.original_module.graph.nodes:
                if node_in_lowered_module.op == "call_function":
                    # Delegated node
                    _insert_op_occurrences_dict(
                        node_name=node_in_lowered_module.name, delegated=True
                    )

    # Calculate the total number of delegated and non-delegated nodes
    num_delegated_nodes = 0
    num_non_delegated_nodes = 0
    for value in op_occurrences_dict.values():
        num_delegated_nodes += value.delegated
        num_non_delegated_nodes += value.non_delegated

    return DelegationInfo(
        num_delegated_nodes=num_delegated_nodes,
        num_non_delegated_nodes=num_non_delegated_nodes,
        num_delegated_subgraphs=delegated_subgraph_counter,
        delegation_by_operator=op_occurrences_dict,
    )
