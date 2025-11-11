# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
from collections import OrderedDict
from math import frexp, isclose, trunc
from typing import Any, Dict, List, Tuple, Type

import torch
from torch import fx
from torch._ops import OpOverload

from torch.fx import GraphModule
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    SourcePartition,
)
from torchao.quantization.pt2e import ObserverOrFakeQuantize
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


def copy_node_metadata(dest_node: fx.Node, src_node: fx.Node) -> None:
    for key in ["nn_module_stack", "stack_trace", "source_fn_stack"]:
        if key in src_node.meta and src_node.meta[key]:
            dest_node.meta[key] = src_node.meta[key]


def quantize_tensor_multiplier(
    requantize_scale_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given requantize_scale_tensor with values in the interval (0, 1),
    produce a pair of tensors (out_multiplier, right_shift) where out_multiplier
    is an int32 tensor representing fixed-point values in the interval [-1, 1),
    and right_shift is an amount to shift right by, so that the floating-point
    multiplication of some int32 input with each value of requantize_scale_tensor:
        result = int32_value * requantize_scale_tensors[i]
    is best approximated by the integer-arithmetic-only code:
        result = RoundingRightShift(FixedPointMultiplication(int32_value,
                                    out_multiplier[i]), right_shift[i])
    """

    # This is identical to C++11 std::round(). The general python round rounds
    # down, and C++ rounds away from zero.
    def round_away_zero(f) -> int:
        r = -0.5 if (f < 0) else 0.5
        return trunc(f + r)

    def quantize_scalar_multiplier(requantize_scale: float) -> Tuple[int, int]:
        significand, exponent = frexp(requantize_scale)
        significand_q31 = int(round_away_zero(significand * (1 << 31)))
        # Handle the special case when the real multiplier was so close to 1
        # that its fixed-point approximation was indistinguishable from 1.
        # We handle this by dividing it by two, incrementing exponent by 1.
        # the right shift amount.
        if significand_q31 == (1 << 31):
            significand_q31 //= 2
            exponent += 1

        # Verify that the decomposition of requantize_scale into significand
        # and exponent is correct.
        reconstructed = significand_q31 / (1 << 31) * pow(2, exponent)
        assert isclose(
            requantize_scale, reconstructed, rel_tol=1e-4, abs_tol=1e-4
        ), "computation of significand and exponent from requantize_scale is not accurate"

        return (significand_q31, exponent)

    # Flatten the input scale tensor so that we can operate on individual values
    orig_shape = requantize_scale_tensor.shape
    flattened_tensor = requantize_scale_tensor.flatten().to(torch.float32)
    out_multiplier = torch.zeros(flattened_tensor.shape, dtype=torch.int32)
    right_shift = torch.zeros(flattened_tensor.shape, dtype=torch.int32)

    # Iterate over the flattened scale tensor and compute the decomposition of
    # each value in scale tensor into significand(out_multiplier) and
    # exponent(right_shift)
    for idx, scale in enumerate(flattened_tensor):
        (si, ex) = quantize_scalar_multiplier(scale)
        out_multiplier[idx], right_shift[idx] = si, ex

    # Reshape the tensors back to the original shape
    out_multiplier = out_multiplier.reshape(orig_shape)
    right_shift = right_shift.reshape(orig_shape)

    return (out_multiplier, right_shift)


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


def create_zero_bias_int32(
    graph_module: GraphModule,
    weight_node: fx.Node,
    bias_scale: float,
) -> fx.Node:
    """
    Creates a zero bias tensor with the shape of weight[0]
    """
    try:
        attr_node = getattr(graph_module, weight_node.target)
    except AttributeError:
        if "val" in weight_node.meta:
            attr_node = weight_node.meta["val"]
        else:
            param_dict = dict(graph_module.named_parameters())
            if weight_node.target in param_dict:
                attr_node = param_dict[weight_node.target]
            else:
                buffer_dict = dict(graph_module.named_buffers())
                if weight_node.target in buffer_dict:
                    attr_node = buffer_dict[weight_node.target]
                else:
                    raise AttributeError(
                        f"Could not find weight tensor for node {weight_node.target}. "
                        f"Node metadata keys: {list(weight_node.meta.keys())}"
                    )

    weight_shape = list(attr_node.shape)
    bias_shape = weight_shape[0]
    new_node = graph_module.graph.call_function(
        torch.ops.aten.full.default,
        ([bias_shape], 0.0),
        {"dtype": torch.int32},
    )

    if "val" in weight_node.meta:
        fake_mode = weight_node.meta["val"].fake_mode
        if fake_mode is not None:
            with fake_mode:
                fake_bias = torch.zeros([bias_shape], dtype=torch.int32)
            new_node.meta["val"] = fake_bias
        else:
            new_node.meta["val"] = torch.zeros([bias_shape], dtype=torch.int32)
        copy_node_metadata(new_node, weight_node)

    return new_node


def get_bias_qparams(
    obs_or_fqs: List[ObserverOrFakeQuantize],
) -> Tuple[torch.Tensor, torch.Tensor]:
    act_scale, _ = obs_or_fqs[0].calculate_qparams()
    weight_scale, _ = obs_or_fqs[1].calculate_qparams()
    bias_scale = act_scale * weight_scale
    bias_zero_point = torch.zeros_like(bias_scale, dtype=torch.int32)
    return bias_scale, bias_zero_point


def get_conv_args(arg, first_val: int) -> List[fx.Node]:
    return arg if len(arg) == 2 else [first_val, arg[0]]


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


def check_out_zero_point_is_min_range(
    out_zero_point: int,
    out_dtype: torch.dtype,
) -> bool:
    """
    Checks if the out_zero_point is the minimum range of the quant type.
    """
    if out_dtype == torch.int8:
        return out_zero_point == -128
    elif out_dtype == torch.int16:
        return out_zero_point == -32768
    elif out_dtype == torch.uint8 or torch.uint16:
        return out_zero_point == 0
    return False
