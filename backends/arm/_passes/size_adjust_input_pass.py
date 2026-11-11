# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Sequence, Set, Type, TypeAlias

import torch
import torch.fx
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    expand_around_channel,
)
from executorch.backends.arm._passes.rewrite_conv_pass import RewriteConvPass
from executorch.backends.arm._passes.rewrite_max_pool2d_pass import RewriteMaxPool2dPass
from executorch.backends.arm._passes.symbolic_value_range import (
    evaluate_symbolic_expr_values,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.backends.arm.tosa.specification import get_context_shape_env
from executorch.backends.transforms.symbolic_shape_utils import materialize_symints
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

Slices: TypeAlias = list[tuple[int, int, int]]
SymIntLike = int | torch.SymInt

conv2d_op = exir_ops.edge.aten.convolution.default
max_pooling_op = exir_ops.edge.aten.max_pool2d.default
avg_pooling_op = exir_ops.edge.aten.avg_pool2d.default
slice_op = exir_ops.edge.aten.slice_copy.Tensor

valid_operators = [conv2d_op, max_pooling_op, avg_pooling_op]


def conv_remainder(
    input_length: SymIntLike, pad: int, dilation: int, weight: int, stride: int
) -> SymIntLike:
    """Returns the remainder of input_length; given the padding, dilation,
    stride, and kernel size.
    """
    return (input_length + 2 * pad - dilation * (weight - 1) - 1) % stride


def has_dynamic_conv_padding(conv_node: torch.fx.Node) -> bool:
    """Return whether rewriting a convolution requires symbolic padding.

    Args:
        conv_node (torch.fx.Node): Convolution node to inspect.

    Returns:
        bool: Whether the convolution needs runtime padding.

    """
    input_node, weight, _, stride_hw, pad_hw, dilation_hw, transposed, _, _ = (
        conv_node.args
    )
    if transposed:
        return True

    input_shape = cast(torch.fx.Node, input_node).meta["val"].shape
    weight_shape = cast(torch.fx.Node, weight).meta["val"].shape
    spatial_rank = len(input_shape) - 2
    strides = expand_around_channel(cast(Sequence[int] | int, stride_hw), spatial_rank)
    pads = expand_around_channel(cast(Sequence[int] | int, pad_hw), spatial_rank)
    dilations = expand_around_channel(
        cast(Sequence[int] | int, dilation_hw), spatial_rank
    )
    shape_env = get_context_shape_env()

    for axis_index, (stride, pad, dilation) in enumerate(zip(strides, pads, dilations)):
        remainder = conv_remainder(
            input_shape[axis_index + 2],
            pad,
            dilation,
            weight_shape[axis_index + 2],
            stride,
        )
        if not isinstance(remainder, torch.SymInt):
            continue
        exact_values = evaluate_symbolic_expr_values(remainder.node.expr, shape_env)
        if exact_values is None or (len(exact_values) != 1 and max(exact_values) != 0):
            return True

    return False


def pooling_remainder(
    input_size: SymIntLike, pad: int, kernel_size: int, stride: int
) -> SymIntLike:
    """Returns the remainder of input_length; given the padding, stride, and
    kernel size.
    """
    return (input_size + 2 * pad - kernel_size) % stride


def _get_pooling_ceil_mode(pooling_node: torch.fx.Node) -> bool:
    ceil_mode_index = (
        5
        if pooling_node.target
        in (
            max_pooling_op,
            exir_ops.edge.aten.max_pool2d_with_indices.default,
        )
        else 4
    )
    return bool(
        pooling_node.args[ceil_mode_index]
        if len(pooling_node.args) > ceil_mode_index
        else False
    )


def _ceil_mode_adds_output(
    input_size: int, pad: int, kernel_size: int, stride: int
) -> bool:
    """Return whether ceil_mode gives one more output element than floor mode.

    PyTorch drops a last ceil_mode window that would start in the right
    padding. Ceil mode then pools over the same windows as floor mode, and the
    input has to be truncated in the same way.

    """
    ceil_post_pad = adjust_pooling_pad_if_needed(
        input_size, kernel_size, stride, pad, True
    )
    floor_post_pad = adjust_pooling_pad_if_needed(
        input_size, kernel_size, stride, pad, False
    )
    return ceil_post_pad != floor_post_pad


def has_dynamic_pooling_padding(pooling_node: torch.fx.Node) -> bool:
    """Return whether rewriting a pool requires symbolic padding.

    Args:
        pooling_node (torch.fx.Node): Pooling node to inspect.

    Returns:
        bool: Whether the pooling operation needs runtime padding.

    """
    input_node = cast(torch.fx.Node, pooling_node.args[0])
    kernel_size = pooling_node.args[1]
    stride = (
        pooling_node.args[2]
        if len(pooling_node.args) >= 3 and pooling_node.args[2]
        else kernel_size
    )
    padding = pooling_node.args[3] if len(pooling_node.args) >= 4 else 0
    ceil_mode = _get_pooling_ceil_mode(pooling_node)
    input_shape = input_node.meta["val"].shape
    kernel_sizes = expand_around_channel(cast(Sequence[int] | int, kernel_size), 2)
    strides = expand_around_channel(cast(Sequence[int] | int, stride), 2)
    pads = expand_around_channel(cast(Sequence[int] | int, padding), 2)
    shape_env = get_context_shape_env()

    for dim, (kernel, stride, pad) in enumerate(
        zip(kernel_sizes, strides, pads), start=2
    ):
        adjusted_pad = adjust_pooling_pad_if_needed(
            input_shape[dim], kernel, stride, pad, bool(ceil_mode)
        )
        if not isinstance(adjusted_pad, torch.SymInt):
            continue
        exact_values = evaluate_symbolic_expr_values(adjusted_pad.node.expr, shape_env)
        if exact_values is None or len(exact_values) != 1:
            return True

    return False


def _greater_than(input: SymIntLike, other: int) -> bool | torch.SymBool:
    """Returns whether an int or SymInt is greater than another value."""
    if isinstance(input, torch.SymInt):
        shape_env = get_context_shape_env()
        exact_values = evaluate_symbolic_expr_values(input.node.expr, shape_env)
        if exact_values is not None:
            return max(exact_values) > other
        value_ranges = shape_env.bound_sympy(input.node.expr)
        return value_ranges.upper > other
    else:
        return input > other


def _get_slice_adjustment(
    remainder: SymIntLike,
    pad: int,
    stride: int,
) -> SymIntLike | None:
    """Return the amount to slice from the end of a conv dimension.

    The required trim is ``max(remainder - pad, 0)``. For symbolic shapes we
    encode that clamp using only integer arithmetic that the TOSA shape
    materializer already supports: a sum of floor-div terms over the possible
    residue classes.

    """
    if not isinstance(remainder, torch.SymInt):
        return remainder - pad if remainder > pad else None

    shape_env = get_context_shape_env()
    exact_values = evaluate_symbolic_expr_values(remainder.node.expr, shape_env)
    if exact_values is not None:
        adjustments = {max(value - pad, 0) for value in exact_values}
        if len(adjustments) == 1:
            exact_adjustment = next(iter(adjustments))
            return exact_adjustment if exact_adjustment > 0 else None

    if pad >= stride - 1:
        return None

    adjustment: SymIntLike | None = None
    for threshold in range(pad + 1, stride):
        term = (remainder + stride - threshold) // stride
        adjustment = term if adjustment is None else adjustment + term

    return adjustment


def get_slices_convolution(conv_node: torch.fx.Node) -> Slices:
    slices: Slices = []

    input_node, weight, _, stride_hw, pad_hw, dilation_hw, transposed, _, _ = (
        conv_node.args
    )
    if transposed:
        return slices
    weight_shape = cast(torch.fx.Node, weight).meta["val"].shape
    input_shape = cast(torch.fx.Node, input_node).meta["val"].shape
    spatial_rank = len(input_shape) - 2

    strides = expand_around_channel(cast(Sequence[int] | int, stride_hw), spatial_rank)
    pads = expand_around_channel(cast(Sequence[int] | int, pad_hw), spatial_rank)
    dilations = expand_around_channel(
        cast(Sequence[int] | int, dilation_hw), spatial_rank
    )

    for axis_index, (stride, pad, dilation) in enumerate(zip(strides, pads, dilations)):
        dim = axis_index + 2
        remainder = conv_remainder(
            input_shape[dim], pad, dilation, weight_shape[dim], stride
        )
        adjustment = _get_slice_adjustment(
            remainder,
            pad,
            stride,
        )
        if adjustment is not None:
            args = (dim, 0, input_shape[dim] - adjustment)
            slices.append(args)

    return slices


def get_slices_pooling(pooling_node: torch.fx.Node) -> Slices:
    slices = []

    input_node = pooling_node.args[0]
    kernel_size = pooling_node.args[1]
    stride = (
        pooling_node.args[2]
        if len(pooling_node.args) >= 3 and pooling_node.args[2]
        else kernel_size
    )
    padding = pooling_node.args[3] if len(pooling_node.args) >= 4 else 0
    ceil_mode = _get_pooling_ceil_mode(pooling_node)

    input_shape = cast(torch.fx.Node, input_node).meta["val"].shape

    kernel_sizes = expand_around_channel(cast(Sequence[int] | int, kernel_size), 2)
    strides = expand_around_channel(cast(Sequence[int] | int, stride), 2)
    pads = expand_around_channel(cast(Sequence[int] | int, padding), 2)

    for dim, (kernel_length, stride_length, pad_size) in enumerate(
        zip(kernel_sizes, strides, pads), start=2
    ):
        # When ceil_mode adds an output element, its window reaches the end of
        # the input, so there is nothing to truncate in this dimension.
        if (
            ceil_mode
            and not isinstance(input_shape[dim], torch.SymInt)
            and _ceil_mode_adds_output(
                input_shape[dim], pad_size, kernel_length, stride_length
            )
        ):
            continue
        remainder = pooling_remainder(
            input_shape[dim], pad_size, kernel_length, stride_length
        )
        if _greater_than(remainder, pad_size):
            adjustment = remainder - pad_size
            args = (dim, 0, input_shape[dim] - adjustment)
            slices.append(args)

    return slices


def get_slices(node: torch.fx.Node) -> Slices:
    """Returns the remainder of input_length; given graph Node."""
    if node.target == conv2d_op:
        return get_slices_convolution(node)
    elif node.target == max_pooling_op or node.target == avg_pooling_op:
        return get_slices_pooling(node)
    else:
        raise ValueError(f"Unsupported node target, was expecting {valid_operators}")


def _has_static_spatial_shape(node: torch.fx.Node) -> bool:
    input_shape = cast(torch.fx.Node, node.args[0]).meta["val"].shape
    return not any(isinstance(dim, torch.SymInt) for dim in input_shape[2:])


def is_valid_operator(node: torch.fx.Node) -> bool:
    if node.target == conv2d_op:
        return True
    elif node.target == max_pooling_op:
        dilation = node.args[4] if len(node.args) >= 5 else 1
        ceil_mode = node.args[5] if len(node.args) >= 6 else False

        # Dilation should be handled first by DecomposeMaxPool2dPass
        if isinstance(dilation, int):
            if dilation > 1:
                raise ValueError(
                    "Expected max_pool2d with dilation = 1, has DecomposeMaxPool2dPass been run?"
                )
        else:
            dilation = cast(list, dilation)
            if dilation[0] > 1 or dilation[1] > 1:
                raise ValueError(
                    "Expected max_pool2d with dilation = [1, 1], has DecomposeMaxPool2dPass been run?"
                )

        # Ceil mode needs the same input adjustment as floor mode where it does
        # not add an output element, see get_slices_pooling. This is only
        # handled for static spatial shapes.
        return not ceil_mode or _has_static_spatial_shape(node)
    elif node.target == avg_pooling_op:
        ceil_mode = node.args[4] if len(node.args) >= 5 else False
        count_include_pad = node.args[5] if len(node.args) >= 6 else True
        divisor_override = node.args[6] if len(node.args) >= 7 else None

        return (
            (not ceil_mode or _has_static_spatial_shape(node))
            and not count_include_pad
            and divisor_override is None
        )

    return False


class SizeAdjustInputPass(ArmPass):
    """Adjusts the input size to Conv2D and Pooling operators. PyTorch allows
    the input and kernel shape to not "match", in which case the remaining
    rows/columns are truncated. However, matching the size is a requirement in
    the TOSA specification. In case the input and kernel shape do not match, the
    following is performed to meet the specification:

      1) The padding is truncated (done in the node visitor)
      2) (if neccessary) The input is truncated (done in this pass)."

    A simple example would be a 2x2 kernel (no padding, stride=2) and a 5x5
    input:

    ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐
    │ X │ X │   │   │   │    │   │   │ X │ X │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │ X │ X │   │   │   │    │   │   │ X │ X │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │ -> │   │   │   │   │   │ -> │ X │ X │   │   │   │ ->
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │    │   │   │   │   │   │    │ X │ X │   │   │   │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │    │   │   │   │   │   │    │   │   │   │   │   │
    └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘
         First pass               second pass              third pass

    ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐
    │   │   │   │   │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │ X │ X │   │ -> │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │ X │ X │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │    │ - │ - │ - │ - │ - │
    └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘
         Fourth pass            Unvisited cells

    Cells that are never visited are marked with `-` and are never considered
    when the kernel traverses over the input, hence they can be removed.

    To match the shape of the kernel (and all parameters) with the input, a
    slice op is inserted to remove the remaining edges (rows and columns) of the
    input.

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        RewriteConvPass,
        RewriteMaxPool2dPass,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if not is_valid_operator(node):
                continue

            target_node = cast(torch.fx.Node, node)
            slice_args = get_slices(target_node)

            if len(slice_args) == 0:
                continue

            parent_node = node.args[0]
            with graph_module.graph.inserting_before(node):
                # ``Graph.create_node`` rejects raw SymInts in
                # call_function args. Aggregate every slice arg across
                # all entries and lift via a single ``materialize_symints``
                # call -- the helper walks the graph once for producer
                # discovery, so a single call amortises that cost and lets
                # symints with shared sub-expressions get hash-consed into
                # one subgraph. Plain ints pass through unchanged.
                flat = [a for args in slice_args for a in args]
                materialized = iter(materialize_symints(graph, flat))
                # Pop exactly len(args) values from the materialized iterator
                # and pack them back into a tuple -- regroups the flat list
                # of lifted values into the original (dim, start, end) shape.
                lifted_slice_args = [
                    tuple(next(materialized) for _ in args) for args in slice_args
                ]

                last_node = cast(torch.fx.Node, parent_node)
                for args in lifted_slice_args:
                    slice_node = create_node(
                        graph,
                        slice_op,
                        (last_node,) + args,
                        from_node=node,
                    )
                    last_node = slice_node
                node.replace_input_with(cast(torch.fx.Node, parent_node), last_node)
                modified = True

        if modified:
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
