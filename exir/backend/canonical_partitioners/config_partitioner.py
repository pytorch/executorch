# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from executorch.exir.backend.backend_details import ExportedProgram
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from torch.fx.passes.infra.partitioner import Partition


def format_target_name(target_name: str) -> str:
    """
    We remove the dialect name space from the target name. We generally
    do not care for the op dialect specific name space ("aten.", "quantized_decomposed.")
    but rather the op itself. Se remove the dialect-specific name space from the
    name and return the op name itself
    """
    names = target_name.split(".")
    if len(names) > 2:
        names.pop(0)

    return ".".join(names)


class PartitionerConfig(ABC):
    """
    Class used to represent a PartitionerConfig.

    PartitionerConfig is used by config-based partitioner to partition identify
    nodes to be delegated. User overrides the methods:
        - target_name
        - check_constraints
        - get_partition
        - get_original_aten

    The Config-Based Partitioner then uses these overridden methods to find nodes
    which match target_name, check_constraints, and if true, returns the partition
    (list of nodes) which represent the node and its dependencies. get_original_aten
    is used to halt decomposition to edge_dialect if the node can be delegated by
    the specified backend.
    """

    @classmethod
    @property
    @abstractmethod
    def target_name(cls) -> str:
        """
        Target name for this partitioner config. When the Config-Based Partitioner
        encounters a node with a matching target name, it uses this config's methods to
        checks the constraints of this node and get all of its dependencies.
        the target name is formatted to remove the dialect-specific name space.
        i.e. linear.default
        """
        pass

    @abstractmethod
    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        Takes in a node and returns true if the node is partitionable.

        Args:
            node: Node to be partitioned
            ep: Exported program of the graph module
        Returns:
            True or False whether this node is partitionable
        """
        pass

    @abstractmethod
    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        """
        Returns the original aten dialect op, this is for to_edge_transform_and_lower
        API, so that this config can be used to stop decomposition of this original
        aten op
        """
        pass

    @abstractmethod
    def get_partition(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        Returns the partitioned nodes from get_node_and_deps, but also labels them
        with the name of the PartitionerConfig class which return this set of nodes.

        Returns an empty list of the node and deps do not satisfy the checked constraints
        """
        pass


class ConfigerationBasedPartitioner(Partitioner):
    def __init__(
        self,
        delegation_spec: DelegationSpec,
        partitioner_configs: Iterable[PartitionerConfig],
    ):
        """
        Configeration based partitioner. We supply the partitioner with a set of configerations
        which describe the node type, constraints, and any dependencies required to be partitioned
        with the node. We use the configerations to partition the graph module.
        """
        super().__init__()
        # Initialize partitioner configs map {"target_name": PartitionerConfig}
        self.target_partitioner_configs: Dict[str, PartitionerConfig] = {}
        for config in partitioner_configs:
            target_name = config.target_name
            if target_name in self.target_partitioner_configs:
                other_config = self.target_partitioner_configs[target_name]
                raise RuntimeError(
                    f"PartitionerConfig: {config} and {other_config} have the same target_name: {target_name}"
                )
            else:
                self.target_partitioner_configs[target_name] = config

        self.delegation_spec = delegation_spec

    def ops_to_not_decompose(
        self,
        ep: ExportedProgram,
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        def filter_fn(node: torch.fx.Node) -> bool:
            """
            The partitioner configs we initialize with have check_constraints function,
            to determine if this op is indeed partitionable. We grab the check_constraint
            function of this op from the config and use it to filter.
            """
            if node.op != "call_function":
                return False
            target_name = format_target_name(node.target.__name__)  # pyre-ignore

            if target_name in self.target_partitioner_configs:
                config = self.target_partitioner_configs[target_name]
                # only filter_fn if config has original_aten
                if config.get_original_aten():
                    return self.target_partitioner_configs[
                        target_name
                    ].check_constraints(node, ep)

            return False

        # Get list of original aten targets which we do not want to decomp
        do_not_decomp = []
        for node_config in self.target_partitioner_configs.values():
            original_aten = node_config.get_original_aten()
            if original_aten is not None:
                do_not_decomp.append(original_aten)

        return (do_not_decomp, filter_fn)

    def get_matched_nodes_from_configs(
        self, ep: ExportedProgram
    ) -> List[List[torch.fx.Node]]:
        # gather supported nodes
        matched_nodes = []
        gm = ep.graph_module
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target = format_target_name(node.target.__name__)
                if target in self.target_partitioner_configs:
                    node_config = self.target_partitioner_configs[target]
                    if node_config.check_constraints(node, ep):
                        matched_nodes.append(node_config.get_partition(node, ep))

        return matched_nodes

    def generate_partitions(self, ep: ExportedProgram) -> List[Partition]:
        matched_nodes = self.get_matched_nodes_from_configs(ep)
        # create partitions
        partitions = generate_partitions_from_list_of_nodes(
            ep.graph_module,
            matched_nodes,
        )
        return partitions

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partitions = self.generate_partitions(exported_program)

        # tag nodes
        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partitions:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                if "delegation_tag" in node.meta:
                    raise RuntimeError(
                        f"Partitioner Erro found node {node} in partition {node.meta['delegation_tag']} and partition {delegation_tag}"
                    )
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
