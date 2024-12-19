# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


# This file contains all the helper utility functions.

from itertools import zip_longest
from math import frexp, isclose, trunc
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.fx

from executorch.exir.dialects._ops import ops as exir_ops
from torch.utils._pytree import tree_flatten


# Return the output node of the graph
def get_output_node(graph: torch.fx.Graph) -> torch.fx.Node:
    assert graph is not None, "Cannot get output of an empty graph"
    output_node = next(iter(reversed(graph.nodes)))
    assert (
        output_node and output_node.op == "output" and len(output_node.args) == 1
    ), "Failed to find output node"
    return output_node


# Return true if the node is part of the flattened output
def is_node_in_flattened_output(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    output_node = get_output_node(graph)
    return node in tree_flatten(output_node.args[0])[0]


# Returns a list with placeholders/inputs
def get_placeholders(graph: torch.fx.Graph) -> List[torch.fx.Node]:
    return list(filter(lambda x: x.op == "placeholder", graph.nodes))


# Return the shape of the incoming node.
def get_shape(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node
) -> Union[torch.Size, None]:
    """
    Return the shape of the tensor correspnding to node. If the node has a
    tensor spec, return the shape from the metadata. If the node is a param,
    return it shape. Otherwise return None.
    """
    try:
        # Case 1. node is a scalar (this pass happens before tensorization)
        if isinstance(node, (float, int, bool)):
            return torch.Size([1])
        # Case 2. node has TensorSpec metadata
        fake_tensor = node.meta.get("val")
        if fake_tensor is not None:
            return fake_tensor.shape
        # Case 3. node holds a param
        if node.op == "get_attr":
            attr_node = getattr(graph_module, node.target)
            return attr_node.shape
        # Default: return None
        return None
    except RuntimeError:
        return None


# Return true if shape_2 can be broadcasted to shape_1
def broadcastable(shape_1: Sequence[int], shape_2: Sequence[int]) -> bool:
    """
    Check if 'shape_2' can be broadcasted to 'shape_1'. The broadcast is
    feasible if:
    (1) shape_2 does not have higher dimensionality than shape_1;
    (2) the value at each dimension of shape_2 is either the same as shape_1 or 1;
    (3) shape_1 or shape_2 is empty.
    """
    return (
        not shape_1
        or not shape_2
        or all(
            x == y or y == 1 or y is None
            for x, y in zip_longest(shape_1[::-1], shape_2[::-1])
        )
    )


# Return a chain of nodes with target in op_targets
def get_cascaded_ops(
    nodes: List[torch.fx.Node],
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    op_targets: Iterable[Union[Callable[..., Any], str]],
) -> Sequence[torch.fx.Node]:
    """
    'nodes' contains a chain of ops with target in 'op_targets'. Extend that chain
    by one if nodes[-1] has a single user with its op target in 'op_targets'.
    """
    cur = nodes[-1]
    users = list(cur.users.keys())
    # Assert that (a) there is only one user of cur, and (b) that user is
    # one of the op in op_targets.
    if len(users) == 1 and users[0].target in op_targets:
        nodes.append(users[0])
        # Recursively find the chain starting at the user
        return get_cascaded_ops(nodes, op_targets)

    return nodes


# Capture the effect of transpose op on incoming dimension order
def get_transposed_dims(node: torch.fx.Node, dims: List[int]) -> List[int]:
    """
    Given a transpose node, and the incoming dimension ordering of the input
    tensor to the transpose node, return the net effect of transpose op on the
    dimension order.
    """
    assert node.target == exir_ops.edge.aten.transpose_copy.int
    # Assert that the dims is not empty
    assert dims is not None
    dim_len = len(dims)
    # Get dim0 and dim1 from the transpose op args
    transpose_dims0 = node.args[1]
    transpose_dims1 = node.args[2]
    assert isinstance(transpose_dims0, int)
    assert isinstance(transpose_dims1, int)
    dim0 = transpose_dims0 if transpose_dims0 >= 0 else transpose_dims0 + dim_len
    dim1 = transpose_dims1 if transpose_dims1 >= 0 else transpose_dims1 + dim_len
    # Perform transpose on dimmension ordering (dims)
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return dims


# Capture the effect of permute op on incoming dimension order
def get_permuted_dims(node: torch.fx.Node, dims: Optional[List[int]]) -> List[int]:
    """
    Given a permute node, and the incoming dimension ordering of the input
    tensor to the permute node, return the net effect of permute op on the
    dimension order.
    """
    assert node.target == exir_ops.edge.aten.permute_copy.default
    # Permute each index of the dimension ordering (dims)
    permute_dims = node.args[1]
    assert isinstance(permute_dims, List)
    assert all(isinstance(x, int) for x in permute_dims)
    # If the dims is empty, we can simply return the permute order
    if not dims:
        return permute_dims
    dims = [dims[x] for x in permute_dims]
    return dims


# Return the tensor of buffer/parameter op
def get_tensor_from_attr(
    graph_module: torch.fx.GraphModule, node: Optional[torch.fx.Node]
) -> Optional[torch.Tensor]:
    """
    For an input node that is a named buffer or parameter, return
    the underlying tensor.
    """
    if node is None:
        return None
    assert node.op == "get_attr"
    return getattr(graph_module, node.target)


def is_node_with_op(node: torch.fx.Node, op: str) -> bool:
    """
    Return true if the incoming node has the given op type
    """
    return node.op == op


def count_users_with_target_op_type(
    nodes: Iterable[torch.fx.Node],
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    op_target: Union[Callable[..., Any], str],
) -> int:
    """
    Given a set of nodes and a node target type `op_target`, iterate over all
    the users of nodes, and return the total number of users with target
    op_target.
    """

    def contributions_per_node(
        node: torch.fx.Node,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        op_target: Union[Callable[..., Any], str],
    ) -> int:
        return [use.target for use in node.users if use.op == "call_function"].count(
            op_target
        )

    return sum([contributions_per_node(node, op_target) for node in nodes])


def contains_node_with_matching_target(
    nodes: Iterable[torch.fx.Node],
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    op_target: Union[Callable[..., Any], str],
) -> bool:
    """
    Given a list of nodes, return true if any node in the list has target
    'op_target'.
    """
    return any(node.target == op_target for node in nodes)


def is_quantized_tensor(x: torch.Tensor) -> bool:
    """
    Return true if the tensor x is quantized
    """
    return x.is_quantized


def get_scale(x: torch.Tensor) -> torch.Tensor:
    """
    Return the scale of a quantized tensor as a float32 tensor.
    """
    return (
        x.q_per_channel_scales().to(torch.float32)
        if x.qscheme() == torch.per_channel_affine
        else torch.tensor([x.q_scale()], dtype=torch.float32)
    )


def get_zero_point(x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """
    Return the zero point of a quantized tensor as int32 tensor.
    """
    # If x was quantized per-tensor, simply create a tensor out of the scalar
    # zero_point, and return it.
    if x.qscheme() == torch.per_tensor_affine:
        return torch.tensor([x.q_zero_point()], dtype=torch.int32)
    # If x was quantized per-channel, check if the zero_point is all zeros. If
    # so, then we can compress the zero_point tensor to a scalar.
    assert x.qscheme() == torch.per_channel_affine, "Unhandled quantization scheme"
    zero_point = x.q_per_channel_zero_points().to(torch.int32)
    return (
        torch.tensor([zero_point[0]], dtype=torch.int32)
        if reduce and all(zero_point == zero_point[0])
        else zero_point
    )


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
    # pyre-fixme[2]: Parameter must be annotated.
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
