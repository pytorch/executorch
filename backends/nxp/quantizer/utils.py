# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type

import torch
from torch import fx
from torch._ops import OpOverload
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    SourcePartition,
)
from torchao.quantization.pt2e import ObserverOrFakeQuantize
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


def is_annotated(nodes: List[fx.Node]) -> bool:
    annotated = False
    for node in nodes:
        annotated = annotated or (
            Q_ANNOTATION_KEY in node.meta and node.meta[Q_ANNOTATION_KEY]._annotated
        )
    return annotated


def no_outside_users(fused_partition) -> bool:
    """
    Checks if each partition other than the last does not have any outside users.
    """
    for source_partition in fused_partition[:-1]:
        if len(source_partition.output_nodes) != 1:
            return False
        if len(source_partition.output_nodes[0].users) != 1:
            return False
    return True


def get_bias_qparams(
    obs_or_fqs: List[ObserverOrFakeQuantize],
) -> Tuple[torch.Tensor, torch.Tensor]:
    act_scale, _ = obs_or_fqs[0].calculate_qparams()
    weight_scale, _ = obs_or_fqs[1].calculate_qparams()
    bias_scale = act_scale * weight_scale
    bias_zero_point = torch.zeros_like(bias_scale, dtype=torch.int32)
    return bias_scale, bias_zero_point


def get_aten_node_target_partitions(
    graph: torch.fx.Graph,
    wanted_original_aten_op: List[OpOverload],
):
    """
    Args:
        graph: The graph we want to partition
        wanted_original_aten_op: List of original_aten ops (OpOverload)

    Returns:
        Dictionary mapping aten ops that were given to a list of SourcePartitions
        that correspond to the list of nodes that were decomposed from the given
        aten ops.
    """
    modules: Dict[Type, Dict[str, List[torch.fx.Node]]] = {}

    for node in graph.nodes:
        # The metadata source_fn should contain a tuple of a unique name for the
        # source, and the source function if the node is decomposed from a
        # function, or the type of module if the node is decomposed from a leaf
        # module
        # TODO(matthiascremon): look into ways to avoid using source_fn_stack
        if (source_fn_st := node.meta.get("source_fn_stack")) is None:
            continue

        source_fn = source_fn_st[-1]
        if node.target not in wanted_original_aten_op:
            continue

        diff_modules = modules.setdefault(source_fn[1], {})
        partition = diff_modules.setdefault(node.name, [])
        partition.append(node)

    def make_partition(
        nodes: List[torch.fx.Node], module_type: Type
    ) -> SourcePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and arg not in nodes:
                    input_nodes.add(arg)

            if node.op == "get_attr":
                params.add(node)

            for user in node.users.keys():
                if user not in nodes:
                    output_nodes.add(node)

        return SourcePartition(
            nodes,
            module_type,
            list(input_nodes),
            list(output_nodes),
            list(params),  # type: ignore[arg-type]
        )

    ret: Dict[Type[Any], List[SourcePartition]] = {}

    for k, v in modules.items():
        ret[k] = [make_partition(partition, k) for partition in v.values()]

    return ret


def _partitions_sequential(partitions: Tuple[SourcePartition]) -> bool:
    prev_partition = None
    for partition in partitions:
        if prev_partition is not None and not check_subgraphs_connected(
            prev_partition, partition
        ):
            return False
        prev_partition = partition
    return True


def find_sequential_partitions_aten(
    gm: torch.fx.GraphModule,
    partition_types: List[Any],
):
    typed_partitions: OrderedDict[Any, List[SourcePartition]] = OrderedDict()
    for partition_type in partition_types:
        partitions = get_aten_node_target_partitions(gm.graph, [partition_type])
        typed_partitions[partition_type] = list(
            itertools.chain.from_iterable(partitions.values())
        )

    typed_partitions_list = list(typed_partitions.values())
    fusion_candidates = itertools.product(*typed_partitions_list)
    fused_partitions = []
    for candidate in fusion_candidates:
        if _partitions_sequential(candidate):
            fused_partitions.append(candidate)
    return fused_partitions
