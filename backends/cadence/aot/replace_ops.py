# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This file contains all the functions that replace one op with another in the
# graph.

# pyre-unsafe

import logging
import math
import operator
from operator import neg
from typing import cast, Dict, Optional, Sequence

import torch
import torch.fx
from executorch.backends.cadence.aot.compiler_utils import quantize_tensor_multiplier
from executorch.backends.cadence.aot.fuse_ops import (
    FuseCascadedTransposeOrPermuteOps,
    FuseCascadedViewOps,
)
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
    RemoveOrReplacePassInterface,
)
from executorch.backends.cadence.aot.remove_ops import RemoveNopSelectOpPass
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult

# A map to represent ops that:
# (a) are functionally equivalent; and
# (b) have identical arguments
# An op whose target is 'key' in this dict can be replaced by the functionally euivalent
# op whose target is 'value'. The replacement would just involve changing the op target.
functionally_equivalent_op_targets: Dict[EdgeOpOverload, EdgeOpOverload] = {
    exir_ops.edge.aten.relu_.default: exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.unsafe_split.Tensor: exir_ops.edge.aten.split_copy.Tensor,
}


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceLogicalNotBooleanWhereWithWherePass(RemoveOrReplacePassInterface):
    """
    A where op with a logical_not and a boolean tensor can be replaced
    by a where op with flipped inputs and the initial boolean tensor.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.where.self]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # If the first arg is not a logical_not, bail.
        if not isinstance(node.args[0], torch.fx.Node):
            return False

        logical_not_node = cast(torch.fx.Node, node.args[0])
        if logical_not_node.target != exir_ops.edge.aten.logical_not.default:
            return False

        # Get the first arg node and its input
        if not isinstance(logical_not_node.args[0], torch.fx.Node):
            return False

        logical_not_input_node = cast(torch.fx.Node, logical_not_node.args[0])

        # If the logical_not input is not a boolean tensor, bail.
        if logical_not_input_node.meta["val"].dtype != torch.bool:
            return False

        # Replace the where op with another one, flipping the inputs and using the boolean
        # tensor from logical_not.
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.where.self,
                args=(logical_not_input_node, node.args[2], node.args[1]),
            )
            new_node.meta = node.meta
        # Replace all the uses
        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceSafeSoftmaxWithSoftmax(RemoveOrReplacePassInterface):  # keep
    """
    Replace _safe_softmax with _softmax
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [torch.ops.aten._safe_softmax.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Add False for the half_to_float argument of softmax
        softmax_args = tuple(list(node.args) + [False])

        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                torch.ops.aten._softmax.default,
                args=softmax_args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplacePT2QuantWithCadenceQuantPass(RemoveOrReplacePassInterface):
    """
    Replace the pt2 quantization ops with cadence quantization ops.
    We do not link kernels to the PT2 quantization ops, so we need to
    replace them with cadence ops at all optimization levels.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        ns = exir_ops.edge if isinstance(node.target, EdgeOpOverload) else torch.ops
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                ns.cadence.quantize_per_tensor.default,
                args=node.args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplacePT2DequantWithCadenceDequantPass(RemoveOrReplacePassInterface):
    """
    Replace the pt2 dequantization ops with cadence dequantization ops.
    We do not link kernels to the PT2 quantization ops, so we need to
    replace them with cadence ops at all optimization levels.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        ns = exir_ops.edge if isinstance(node.target, EdgeOpOverload) else torch.ops
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                ns.cadence.dequantize_per_tensor.default,
                args=node.args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceSqueezeAndUnsqueezeWithViewPass(RemoveOrReplacePassInterface):
    """
    When the shape is static, replace squeeze_copy and unsqueeze_copy ops with
    view_copy op
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.squeeze_copy.default,
            exir_ops.edge.aten.squeeze_copy.dim,
            exir_ops.edge.aten.squeeze_copy.dims,
            exir_ops.edge.aten.unsqueeze_copy.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the output tensor shape
        out_shape = node.meta["val"].shape

        # Bail out if any dim is not an int (dynamic shape)
        for dim in list(out_shape):
            if not isinstance(dim, int):
                return False

        # Replace with view op with the new shape
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(node.args[0], list(out_shape)),
            )
            # Do not remove the metadata copy!
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceFunctionallyEquivalentOpTargets(RemoveOrReplacePassInterface):
    """
    Replace an op with a functionally equivalent op by just switching the op
    target, but without incurring any change to the op args.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return list(functionally_equivalent_op_targets.keys())

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        assert isinstance(node.target, EdgeOpOverload)
        target_op = functionally_equivalent_op_targets[node.target]
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                target_op,
                args=node.args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)

        # RemoveOrReplacePassInterface calls eliminate_dead_code, but this doesn't
        # remove impure nodes (nodes which have side effects). Not sure if that is
        # generally safe, so instead of modifying the interface, just erasing
        # these nodes for this pass.
        node.graph.erase_node(node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceSelectWithViewOpPass(RemoveOrReplacePassInterface):
    """
    If the size along the select dim is 1, then the select op can be replaced
    by view op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.select_copy.int]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the input tensor and shapes
        in_tensor_node = node.args[0]
        assert isinstance(in_tensor_node, torch.fx.Node)
        in_shape = in_tensor_node.meta["val"].shape
        out_shape = node.meta["val"].shape

        # Get the select dimension
        select_dim = node.args[1]
        assert isinstance(select_dim, int)
        select_dim = select_dim if select_dim >= 0 else select_dim + len(in_shape)

        if in_shape[select_dim] == 1:
            # Replace with view op with the new shape
            with node.graph.inserting_before(node):
                new_node = node.graph.call_function(
                    exir_ops.edge.aten.view_copy.default,
                    args=(node.args[0], list(out_shape)),
                )
                # Important to copy metadata
                new_node.meta = node.meta
            node.replace_all_uses_with(new_node)
            return True

        return False


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceMMWithAddMMPass(RemoveOrReplacePassInterface):
    """
    This pass replaces mm with addmm by introducing a zero bias.
    mm is not supported, so this is an opt_level=0 pass.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.mm.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # The mm op has two args: input, mat2
        assert len(node.args) == 2
        X, mat2 = node.args
        assert isinstance(X, torch.fx.Node)
        assert isinstance(mat2, torch.fx.Node)

        # Create a zero bias tensor, and insert it as a graph buffer before the
        # current node
        mat2_tensor = mat2.meta["val"]
        bias_size = mat2_tensor.size(1)

        with node.graph.inserting_before(node):
            zero_bias = node.graph.call_function(
                exir_ops.edge.aten.full.default,
                args=([bias_size], 0.0),
                kwargs={"dtype": torch.float32},
            )
            zero_bias.meta = node.meta

        # Replace mm with addmm
        new_args = (zero_bias, X, mat2)
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.addmm.default,
                args=new_args,
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceAddMMWithLinearPass(RemoveOrReplacePassInterface):
    """
    This pass replaces addmm with linear op.

    AddMM computes: beta*bias + alpha*mm(mat1, mat2)
    Linear computes: mat1 @ weight.T + bias

    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.addmm.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # The addmm op has three concrete args: bias, mat1, mat2
        assert len(node.args) >= 3
        (bias, mat1, mat2) = node.args[0:3]

        # The other two args are optional scale args
        beta = float(node.kwargs.get("beta", 1.0))
        alpha = float(node.kwargs.get("alpha", 1.0))

        bias, mat1, mat2 = cast(
            tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node],
            (bias, mat1, mat2),
        )

        graph = node.graph

        fit_bias = beta == 1.0
        fit_mat2 = False

        # Handle transpose: if mat2 is a transpose op, extract the original tensor
        transposed_mat2 = False
        if (
            mat2.op == "call_function"
            and mat2.target == exir_ops.edge.aten.transpose_copy.int
        ):
            # mat2 is already transposed, so we use the input to the transpose
            mat2 = cast(torch.fx.Node, mat2.args[0])
            transposed_mat2 = True
            fit_mat2 = alpha == 1.0

        if not (fit_bias and fit_mat2):
            return False

        # Multiply bias by beta if needed
        if beta != 1.0:
            # Create a scaled bias using element-wise multiplication in the graph
            with graph.inserting_before(node):
                beta_scalar = graph.call_function(
                    exir_ops.edge.aten.full.default,
                    args=([1], beta),
                    kwargs={"dtype": torch.float32},
                )
                beta_scalar.meta = node.meta
                bias = graph.call_function(
                    exir_ops.edge.aten.mul.Tensor,
                    args=(bias, beta_scalar),
                )

                # Metadata copy important
                bias.meta = node.meta

        # Multiply mat2 by alpha if needed
        if alpha != 1.0:
            with graph.inserting_before(node):
                alpha_scalar = graph.call_function(
                    exir_ops.edge.aten.full.default,
                    args=([1], alpha),
                    kwargs={"dtype": torch.float32},
                )
                alpha_scalar.meta = node.meta
                mat2 = graph.call_function(
                    exir_ops.edge.aten.mul.Tensor,
                    args=(mat2, alpha_scalar),
                )

                # Metadata copy important
                mat2.meta = node.meta

        # Transpose mat2 if it wasn't already transposed
        if not transposed_mat2:
            with graph.inserting_before(node):
                mat2 = graph.call_function(
                    exir_ops.edge.aten.transpose_copy.int,
                    args=(mat2, -1, -2),
                )

                # Metadata copy important
                mat2.meta = node.meta

        # Construct the linear node: linear(input, weight, bias)
        # linear computes: input @ weight.T + bias
        linear_args = (mat1, mat2, bias)
        with graph.inserting_before(node):
            linear_node = graph.call_function(
                exir_ops.edge.aten.linear.default,
                args=linear_args,
            )

            # Metadata copy important
            linear_node.meta = node.meta

        # Replace all uses of the addmm op with linear op
        node.replace_all_uses_with(linear_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplacePermuteWithTransposePass(RemoveOrReplacePassInterface):
    """
    Replace permute op with transpose if the permutation is only along
    two dimensions.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.permute_copy.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the old dim and new dim order
        in_tensor = node.args[0]
        assert isinstance(in_tensor, torch.fx.Node)
        in_shape = in_tensor.meta["val"].shape
        old_dims = tuple(range(len(in_shape)))
        new_dims = cast(Sequence[int], node.args[1])

        # Compute the number of positions in which the old and new order differ
        diff = [od for od, nd in zip(old_dims, new_dims) if od != nd]

        # If the difference is zero, replace with identity (just the input)
        if len(diff) == 0:
            node.replace_all_uses_with(in_tensor)
            return True

        # If the difference is in two dimensions, we can replace this permute op
        # with transpose op.
        if len(diff) == 2:
            with node.graph.inserting_before(node):
                new_node = node.graph.call_function(
                    exir_ops.edge.aten.transpose_copy.int,
                    args=(node.args[0], diff[0], diff[1]),
                )
                new_node.meta = node.meta
            node.replace_all_uses_with(new_node)
            return True

        return False


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceConvolutionOptionalArgsWithConcreteArgsPass(RemoveOrReplacePassInterface):
    """
    Replace optional tensors with concrete tensors. Currently, we
    replace the optional bias tensor with a zero tensor.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.cadence.conv1d.default,
            exir_ops.edge.cadence.conv2d.default,
            exir_ops.edge.cadence.conv3d.default,
            exir_ops.edge.cadence.transposed_convolution.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Check if this is a transposed convolution
        assert isinstance(node.target, EdgeOpOverload)
        is_transposed = (
            node.target == exir_ops.edge.cadence.transposed_convolution.default
        )
        num_expected_args = 9 if is_transposed else 7
        assert len(node.args) == num_expected_args
        # Check if the bias is concrete
        if node.args[2] is not None:
            return False

        # The bias length is the number of out channels.
        out_shape = node.meta["val"].shape
        bias_size = out_shape[1]

        # Create a zero bias tensor
        with node.graph.inserting_before(node):
            zero_bias = node.graph.call_function(
                exir_ops.edge.aten.full.default,
                args=([bias_size], 0.0),
                kwargs={"dtype": torch.float32},
            )
            # Create proper metadata for the zero_bias node
            zero_bias.meta = node.meta
            new_args = list(node.args)
            new_args[2] = zero_bias
            new_args = tuple(new_args)

            new_node = node.graph.call_function(
                # pyre-ignore[6]: Target is a call func, but type is union call func and str
                node.target,
                args=new_args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceRepeatWithCatPass(RemoveOrReplacePassInterface):
    """
    Replace repeat op as successive cat ops along different dimensions.
    repeat is not supported, so this is an opt_level=0 pass.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.repeat.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Extract the input tensor, and the repeats from the args
        in_tensor = node.args[0]
        assert isinstance(in_tensor, torch.fx.Node)
        repeats = cast(Sequence[int], node.args[1])

        # Glean the shapes of input tensor
        in_shape = list(in_tensor.meta["val"].shape)

        # If the size of repeats is more than the dimensionality of the tensor,
        # the output of repeat will be a higher-dimensional tensor. We reshape
        # the input so that it has the same dimensionality as the output tensor.
        diff = len(repeats) - len(in_shape)
        assert (
            diff >= 0
        ), "Repeat arg malformed: expected a repeat along each dimension of input tensor"

        graph = node.graph
        result_node = in_tensor

        if diff > 0:
            # Extend the input shape with 1's along the higher dimensions
            in_shape = ([1] * diff) + in_shape
            # Insert a view op that reshapes the input tensor to have same
            # dimensionality as the output tensor.
            with graph.inserting_before(node):
                result_node = graph.call_function(
                    exir_ops.edge.aten.view_copy.default,
                    args=(in_tensor, in_shape),
                )
                result_node.meta = node.meta
            assert len(repeats) == len(in_shape)

        # Repeat op is nothing but successive cat ops along each dimension.
        for dim, repeat in reversed(list(enumerate(repeats))):
            # We do not need to do anything if repeat factor is 1
            if repeat == 1:
                continue
            cat_arg = [result_node] * repeat
            with graph.inserting_before(node):
                result_node = graph.call_function(
                    exir_ops.edge.aten.cat.default, args=(cat_arg, dim)
                )
                result_node.meta = node.meta

        node.replace_all_uses_with(result_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplacePadWithCatPass(RemoveOrReplacePassInterface):
    """
    Replace constant pad nd op that does padding on outer-most dimension
    with Cat(left_padding_constant_tensor, X, right_padding_constant_tensor)
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.constant_pad_nd.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        assert len(node.args) >= 2
        input_node, orig_padding = node.args[:2]
        assert isinstance(input_node, torch.fx.Node)

        # if there is no padding, this op will be treated in removal pass.
        if not orig_padding:
            return False

        value = 0 if len(node.args) == 2 else node.args[2]

        arg_shape = input_node.meta["val"].shape

        # Convert orig_padding to a list for manipulation
        # pyre-ignore[6]: Argument type
        padding_list = list(orig_padding)
        padding = padding_list + ([0] * (len(padding_list) % 2 != 0))
        assert len(padding) >= 2
        (left_padding_size, right_padding_size) = padding[-2:]
        # Replace only if constant_pad_nd is along the innermost padding dimension.
        if (
            any(x != 0 for x in padding[0:-2])
            or left_padding_size < 0
            or right_padding_size < 0
        ):
            return False

        cat_tensors = []
        dim = len(arg_shape) - len(padding) // 2
        graph = node.graph

        # add left_padding
        if left_padding_size > 0:
            left_padding_shape = (
                arg_shape[:dim] + (left_padding_size,) + arg_shape[dim + 1 :]
            )
            with graph.inserting_before(node):
                left_padding_node = graph.call_function(
                    exir_ops.edge.aten.full.default,
                    args=(
                        left_padding_shape,
                        value,
                    ),
                    kwargs={"dtype": torch.float32},
                )
                left_padding_node.meta = node.meta
            cat_tensors.append(left_padding_node)

        # input_node
        cat_tensors.append(input_node)

        # right_padding
        if right_padding_size > 0:
            right_padding_shape = (
                arg_shape[:dim] + (right_padding_size,) + arg_shape[dim + 1 :]
            )
            with graph.inserting_before(node):
                right_padding_node = graph.call_function(
                    exir_ops.edge.aten.full.default,
                    args=(
                        right_padding_shape,
                        value,
                    ),
                    kwargs={"dtype": torch.float32},
                )
                right_padding_node.meta = node.meta
            cat_tensors.append(right_padding_node)

        assert len(cat_tensors) == 1 + (left_padding_size > 0) + (
            right_padding_size > 0
        )

        new_args = (cat_tensors, dim)
        with graph.inserting_before(node):
            new_node = graph.call_function(
                exir_ops.edge.aten.cat.default,
                args=new_args,
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceConstantPadNdWithSlicePass(RemoveOrReplacePassInterface):
    """
    Replace constant pad nd op that does padding on outer-most dimension
    with exir_ops slice(left_padding_constant_tensor, X, right_padding_constant_tensor)
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.constant_pad_nd.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        assert len(node.args) >= 2
        input_node = node.args[0]
        orig_padding = cast(Sequence[int], node.args[1])
        assert isinstance(input_node, torch.fx.Node)

        # if there is no padding, this op will be treated in removal pass.
        if not orig_padding:
            return False

        padding = list(orig_padding) + ([0] * (len(orig_padding) % 2 != 0))
        assert len(padding) >= 2

        # pyre-ignore[6]
        (start, diff) = map(neg, padding[-2:])
        # Replace only if constant_pad_nd is along the innermost padding dimension.
        if any(x != 0 for x in padding[0:-2]) or start < 0 or diff < 0:
            return False

        arg_shape = input_node.meta["val"].shape
        dim = len(arg_shape) - len(padding) // 2
        stop = arg_shape[dim] - diff
        assert start <= stop

        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.slice.Tensor,
                args=(input_node, dim, start, stop),
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


# Make that pass runnable standalone at opt level 0.
@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAtenConvolutionWithCadenceConvolutionPass(RemoveOrReplacePassInterface):
    """
    Replace aten convolution op with jarvis-specific convolution op, since the
    aten version is not supported by jarvis.
    Also remove convolution stride if the output size along the strided dimension
    is 1. We can enable more transformations (e.g., conv -> linear replacement)
    for unit-stride convolutions.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.convolution.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # There must be 9 total args.
        if len(node.args) != 9:
            return False

        # Unpack the args
        (
            in_tensor,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args

        # Cast to appropriate types
        stride = cast(Sequence[int], stride)
        padding = cast(Sequence[int], padding)
        dilation = cast(Sequence[int], dilation)
        output_padding = cast(Sequence[int], output_padding)

        # Currently we only handle conversion to conv1d, conv2d, and conv3d, therefore
        # verify that the stride, padding, dilation, and output_padding have
        # len <=3.
        if not (
            (len(stride) == len(padding) == len(dilation) == len(output_padding) == 1)
            or (
                len(stride) == len(padding) == len(dilation) == len(output_padding) == 2
            )
            or (
                len(stride) == len(padding) == len(dilation) == len(output_padding) == 3
            )
        ):
            return False

        # Determine if this is 1D, 2D, or 3D convolution based on parameter lengths
        if transposed:
            target = exir_ops.edge.cadence.transposed_convolution.default
        elif len(stride) == 1:
            target = exir_ops.edge.cadence.conv1d.default
        elif len(stride) == 2:
            target = exir_ops.edge.cadence.conv2d.default
        else:  # len(stride) == 3
            target = exir_ops.edge.cadence.conv3d.default

        with node.graph.inserting_before(node):
            if transposed:
                # Flip the height and width dimensions of weight, since we apply a
                # gather stencil. Also, the first two dimensions of weight must be
                # transposed/interchanged.
                assert isinstance(weight, torch.fx.Node)
                transposed_weight = node.graph.call_function(
                    exir_ops.edge.aten.transpose_copy.int,
                    args=(weight, 0, 1),
                )
                transposed_weight.meta = weight.meta

                # Get the dimension for flip based on weight shape
                weight_dim = len(weight.meta["val"].shape)
                flip_dims = [-1] if weight_dim == 3 else [-1, -2]

                flipped_weight = node.graph.call_function(
                    exir_ops.edge.aten.flip.default,
                    args=(transposed_weight, flip_dims),
                )
                flipped_weight.meta = transposed_weight.meta

                new_args = (
                    in_tensor,
                    flipped_weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    groups,
                    False,
                )
            else:
                # Verify that output_padding is 0.
                if not all(x == 0 for x in output_padding):
                    return False

                # Keep the original stride to maintain correct output dimensions
                new_stride = stride

                new_args = (
                    in_tensor,
                    weight,
                    bias,
                    new_stride,
                    padding,
                    dilation,
                    groups,
                )

            new_node = node.graph.call_function(target, args=new_args)
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceTrivialConvWithLinear(RemoveOrReplacePassInterface):
    """
    In nn.Conv1d, the operand shapes are:
        input - [batch, in_channels, in_length]
        weight - [out_channels, in_channels, weight_length]
        output - [batch, out_channels, out_length]
    When in_length == weight_length, out_length = 1. In this scenario, we can
    view the input as a tensor shaped [batch, K], and weight as a tensor
    shaped [out_channels, K], and replace nn.Conv1d with nn.Linear. This
    optimization can be extended to nn.Conv2d as well, where in_length is a 2d
    image, and weight_length can be replaced with a 2d filter the same shape as
    the image.
    """

    trivial_conv_op_to_linear_op: Dict[EdgeOpOverload, EdgeOpOverload] = {
        exir_ops.edge.cadence.conv1d.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.conv2d.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.conv3d.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor: exir_ops.edge.cadence.quantized_linear.per_tensor,
        exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor: exir_ops.edge.cadence.quantized_linear.per_tensor,
    }

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return list(self.trivial_conv_op_to_linear_op.keys())

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Parse the necessary args of the convolution node. Both convolution
        # and quantized_conv have the same first 8 args. The quantized op has
        # extra args holding at least the zero point and scale of input, weight, bias,
        # and output tensor.
        assert isinstance(node.target, EdgeOpOverload)
        quantized_op = (
            node.target == exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor
            or node.target == exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor
        )
        assert (len(node.args) == 7 and not quantized_op) or (
            len(node.args) >= 12 and quantized_op
        ), "Inconsistent args for convolution"
        (in_tensor, weight, bias, stride, padding, dilation, groups) = node.args[0:7]

        assert isinstance(in_tensor, torch.fx.Node)
        assert isinstance(weight, torch.fx.Node)

        # Glean the shapes of input, weight, and output
        in_shape = in_tensor.meta["val"].shape
        weight_shape = weight.meta["val"].shape
        out_shape = node.meta["val"].shape
        assert None not in {in_shape, weight_shape, out_shape}

        # pyre-ignore[6]: Argument type for iteration
        stride_list = list(stride)
        # pyre-ignore[6]: Argument type for iteration
        padding_list = list(padding)
        # pyre-ignore[6]: Argument type for iteration
        dilation_list = list(dilation)

        # Check the condition under which conv can be replaced by linear: (1) this
        # should not be a depthwise convolution; (2) the padding, stride, and dilation
        # should be standard; (3) The [channels, height, width] of input must match the
        # [channel, kernel_height, kernel_width] of the weight. These conditions would
        # ensure that output height and width are 1, and the convolution can be replaced
        # by linear.
        if (
            groups != 1
            or any(x != 0 for x in padding_list)
            or any(x != 1 for x in stride_list)
            or any(x != 1 for x in dilation_list)
            or (list(in_shape[1:]) != list(weight_shape[1:]))
        ):
            return False

        # Reshape the weight to [out_channels, in_channels * X]
        K = math.prod(weight_shape[1:])

        graph = node.graph

        # Weight is always a Node, so we need a view_copy operation
        with graph.inserting_before(node):
            linear_weight = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(
                    weight,
                    [weight_shape[0], K],
                ),
            )
            linear_weight.meta = node.meta

        # Reshape the input from 3d to 2d tensor
        with graph.inserting_before(node):
            in_view = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(
                    in_tensor,
                    [in_shape[0], K],
                ),
            )
            in_view.meta = node.meta

        # Create the linear node, which multiplies the 2d input and weight
        # tensors, and adds the 1d bias to produce a 2d output.
        if quantized_op:
            (
                in_zero_point,
                weight_zero_point,
                bias_scale,
                out_scale,
                out_zero_point,
            ) = node.args[7:12]
            # If the multiplier and shift tensors are provided, use them.
            if len(node.args) >= 14:
                out_multiplier = node.args[12]
                out_shift = node.args[13]
            # If not, compute them.
            else:
                # pyre-ignore[58]: Division operands
                requantize_scale = bias_scale / out_scale
                (out_multiplier, out_shift) = quantize_tensor_multiplier(
                    requantize_scale
                )
            linear_args = (
                in_view,
                linear_weight,
                bias,
                in_zero_point,
                weight_zero_point,
                out_multiplier,
                out_shift,
                out_zero_point,
                None,
            )
        else:
            linear_args = (in_view, linear_weight, bias)
        with graph.inserting_before(node):
            linear_res = graph.call_function(
                self.trivial_conv_op_to_linear_op[cast(EdgeOpOverload, node.target)],
                args=linear_args,
            )
            linear_res.meta = node.meta

        # Reshape the output of linear from 2d to 3d tensor
        with graph.inserting_before(node):
            out_res = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(linear_res, list(out_shape)),
            )
            out_res.meta = node.meta

        node.replace_all_uses_with(out_res)
        return True


def canonicalize_transposed_dim(dim: int, shape: Sequence[int]) -> int:
    """Canonicalize transpose ops so it gets easier to pattern-match and fuse transpose ops."""
    if dim < 0:
        # Keep transpose dimensions positive.
        dim += len(shape)
    return dim


@register_cadence_pass(CadencePassAttribute(opt_level=3))
class ReplaceConvWithChannelLastConvPass(RemoveOrReplacePassInterface):
    """
    Replace NCHW convolutions with NHWC (channel-last) convolutions by adding
    transpose operations before and after the convolution.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.cadence.conv1d.default,
            exir_ops.edge.cadence.conv2d.default,
            exir_ops.edge.cadence.conv3d.default,
            exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
        ]

    def _transpose_dims(
        self, graph: torch.fx.Graph, node: torch.fx.Node, dim0: int, dim1: int
    ) -> torch.fx.Node:
        """Helper function to transpose dims of a node."""
        shape = node.meta["val"].shape
        dim0, dim1 = (
            canonicalize_transposed_dim(dim0, shape),
            canonicalize_transposed_dim(dim1, shape),
        )
        dim0, dim1 = min(dim0, dim1), max(dim0, dim1)
        transpose_node = graph.call_function(
            exir_ops.edge.aten.transpose_copy.int, (node, dim0, dim1), {}
        )
        transpose_node.meta = node.meta
        return transpose_node

    def _change_nchw_to_nhwc(
        self, graph: torch.fx.Graph, node: torch.fx.Node
    ) -> torch.fx.Node:
        """Convert NCHW format to NHWC format."""
        shape = node.meta["val"].shape
        if len(shape) == 3:
            return self._transpose_dims(graph, node, 1, -1)
        indices = list(range(len(shape)))
        permute_indices = [indices[0]] + indices[2:] + [indices[1]]
        permute_node = graph.call_function(
            exir_ops.edge.aten.permute_copy.default, (node, permute_indices), {}
        )
        permute_node.meta = node.meta
        return permute_node

    def _change_nhwc_to_nchw(
        self, graph: torch.fx.Graph, node: torch.fx.Node
    ) -> torch.fx.Node:
        """Convert NHWC format to NCHW format."""
        shape = node.meta["val"].shape
        if len(shape) == 3:
            return self._transpose_dims(graph, node, 1, -1)
        indices = list(range(len(shape)))
        permute_indices = [indices[0], indices[-1]] + indices[1:-1]
        permute_node = graph.call_function(
            exir_ops.edge.aten.permute_copy.default, (node, permute_indices), {}
        )
        permute_node.meta = node.meta
        return permute_node

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        assert isinstance(node.target, EdgeOpOverload)
        quantized_op = (
            node.target == exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor
        )

        # Check if already in NHWC layout
        if not quantized_op and len(node.args) == 8 and node.args[-1] is True:
            return False

        # Determine the new op target
        if quantized_op:
            new_op = exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor
        else:
            new_op = node.target

        graph = node.graph

        # Get input and weight nodes
        input_node = cast(torch.fx.Node, node.args[0])
        weight_node = cast(torch.fx.Node, node.args[1])

        # Insert transpose operations before the node
        with graph.inserting_before(node):
            # Convert input from NCHW to NHWC
            input_nhwc = self._change_nchw_to_nhwc(graph, input_node)
            # Convert weight from NCHW to NHWC
            weight_nhwc = self._change_nchw_to_nhwc(graph, weight_node)

            # Non-quantized ops need to set the last optional argument to True
            channel_last_arg = [] if quantized_op else [True]

            # Create new args with transposed input/weights
            new_args = (
                (input_nhwc, weight_nhwc)
                + tuple(node.args[2:])
                + tuple(channel_last_arg)
            )

            # Create the new conv operation
            new_conv = graph.call_function(new_op, new_args, node.kwargs)
            new_conv.meta = node.meta

            # Convert output back from NHWC to NCHW
            nchw_output = self._change_nhwc_to_nchw(graph, new_conv)

        # Replace all uses with the final output
        node.replace_all_uses_with(nchw_output)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=3))
class MakeSliceAndCatDimOutermostPass(RemoveOrReplacePassInterface):
    """
    Make the slice/cat dimension the outermost dimension by adding transpose
    operations before and after the slice/cat operation.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.slice_copy.Tensor,
        ]

    def _transpose_dims(
        self, graph: torch.fx.Graph, node: torch.fx.Node, dim0: int, dim1: int
    ) -> torch.fx.Node:
        """Helper function to transpose dims of a node."""
        shape = node.meta["val"].shape
        dim0, dim1 = (
            canonicalize_transposed_dim(dim0, shape),
            canonicalize_transposed_dim(dim1, shape),
        )
        dim0, dim1 = min(dim0, dim1), max(dim0, dim1)
        transpose_node = graph.call_function(
            exir_ops.edge.aten.transpose_copy.int, (node, dim0, dim1), {}
        )
        transpose_node.meta = node.meta
        return transpose_node

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the dimension argument
        dim = cast(int, node.args[1]) if len(node.args) > 1 else 0
        output_shape = node.meta["val"].shape

        # Canonicalize dim to be positive
        if dim < 0:
            dim += len(output_shape)

        # Not needed if dim is already outermost or all dims before it are 1
        if dim == 0 or math.prod(output_shape[:dim]) == 1:
            return False

        graph = node.graph

        with graph.inserting_before(node):
            if node.target == exir_ops.edge.aten.slice_copy.Tensor:
                # Transpose input -> slice with dim=0 -> transpose back
                input_node = cast(torch.fx.Node, node.args[0])
                transposed_input = self._transpose_dims(graph, input_node, dim, 0)

                # Create slice operation with dim=0
                slice_args = (transposed_input, 0) + node.args[2:]
                sliced = graph.call_function(
                    exir_ops.edge.aten.slice_copy.Tensor, slice_args, node.kwargs
                )
                sliced.meta = node.meta

                # Transpose back
                result = self._transpose_dims(graph, sliced, 0, dim)
            else:
                # Cat operation: transpose all inputs -> cat with dim=0 -> transpose back
                cat_inputs = cast(list[torch.fx.Node], node.args[0])
                transposed_inputs = [
                    self._transpose_dims(graph, t, dim, 0) for t in cat_inputs
                ]

                # Create cat operation with dim=0
                catted = graph.call_function(
                    exir_ops.edge.aten.cat.default, (transposed_inputs, 0), node.kwargs
                )
                catted.meta = node.meta

                # Transpose back
                result = self._transpose_dims(graph, catted, 0, dim)

        # Replace all uses with the final result
        node.replace_all_uses_with(result)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceConvWithIm2RowAndLinear(RemoveOrReplacePassInterface):
    """
    Replace convolution where groups=1 with im2row followed by a linear op.
    """

    # A map from the convolution op to the linear op that it should
    # decompose to.
    conv_op_to_linear_op: Dict[EdgeOpOverload, EdgeOpOverload] = {
        exir_ops.edge.cadence.conv1d.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.conv2d.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.conv3d.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor: exir_ops.edge.cadence.quantized_linear.per_tensor,
        exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor: exir_ops.edge.cadence.quantized_linear.per_tensor,
    }

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return list(self.conv_op_to_linear_op.keys())

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the relevant args from convolution node.
        assert isinstance(node.target, EdgeOpOverload)
        quantized_op = (
            node.target == exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor
            or node.target == exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor
        )
        assert (len(node.args) == 7 and not quantized_op) or (
            len(node.args) >= 12 and quantized_op
        ), "Inconsistent args for convolution"
        (in_tensor, weight, bias, stride, padding, dilation, groups) = node.args[0:7]

        assert isinstance(in_tensor, torch.fx.Node)
        assert isinstance(weight, torch.fx.Node)

        # We do not replace depthwise convolution with gemm yet.
        if groups != 1:
            return False

        weight_shape = weight.meta["val"].shape

        # pyre-ignore[6]: Argument type for iteration
        stride_list = list(stride)
        # pyre-ignore[6]: Argument type for iteration
        padding_list = list(padding)
        # pyre-ignore[6]: Argument type for iteration
        dilation_list = list(dilation)

        # If this is a pointwise convolution, im2col will start dominating the
        # runtime. So we call convolution op for this case.
        if (
            all(x == 1 for x in weight_shape[2:])
            and all(x == 1 for x in stride_list)
            and all(x == 0 for x in padding_list)
            and all(x == 1 for x in dilation_list)
        ):
            return False

        # Get the shapes
        out_shape = node.meta["val"].shape
        assert None not in {weight_shape, out_shape}

        # Determine if the convolution is NCHW or NHWC. The NHWC, i.e., the
        # channel_last layout is specified by the channel_last arg of conv
        # op, which is either the last argument (15th) or implicitely False
        # if the op is quantized, or the last argument if not.
        channel_last = (
            node.target == exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor
        )
        # The weight tensor is [out_channels, in_channels, X] for NCHW layout,
        # and [out_channels, X, in_channels] for NHWC layout. Here, X is the
        # kernel_width for conv1d, and X = kernel_height * kernel_width for
        # conv2d. We extract X as the kernel_size for im2row.
        kernel_size = list(weight_shape[1:-1] if channel_last else weight_shape[2:])
        # If the convolution op was quantized, we need the input tensor's
        # zero_point for im2row. Otherwise in_zero_point defaults to a zero
        # tensor.
        in_zero_point = node.args[7] if quantized_op else 0

        # im2row expects every kernel parameter to be 2d. So we extend the
        # parameters for conv1d by prepending their default values.
        stride_2d = ([1] + stride_list) if len(stride_list) == 1 else stride_list
        padding_2d = ([0] + padding_list) if len(padding_list) == 1 else padding_list
        dilation_2d = (
            ([1] + dilation_list) if len(dilation_list) == 1 else dilation_list
        )
        kernel_size = ([1] + kernel_size) if len(kernel_size) == 1 else kernel_size
        # Assert that kernel size does not have a 0
        assert 0 not in kernel_size

        graph = node.graph

        # Create an im2row node with the input. This will create a 2d matrix of
        # shape [out_height*out_weight, X*in_channels]. X is as defined in the
        # comment above.
        im2row_args = (
            in_tensor,
            kernel_size,
            dilation_2d,
            padding_2d,
            stride_2d,
            in_zero_point,
            channel_last,
        )
        with graph.inserting_before(node):
            im2row = graph.call_function(
                exir_ops.edge.cadence.im2row.per_tensor,
                args=im2row_args,
            )
            im2row.meta = node.meta

        # Get the product of the >2 dims of the weight
        K = math.prod(weight_shape[1:])

        # Weight is always a Node, so we need a view_copy operation
        with graph.inserting_before(node):
            linear_weight = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(
                    weight,
                    [weight_shape[0], K],
                ),
            )
            linear_weight.meta = node.meta

        # Create the linear node, which multiplies the 3d input with 2d weight
        # tensors with bias addition. The outermost dimension of the input is
        # the batch size for linear op.
        if quantized_op:
            (
                in_zero_point,
                weight_zero_point,
                bias_scale,
                out_scale,
                out_zero_point,
            ) = node.args[7:12]
            # If the multiplier and shift tensors are provided, use them.
            if len(node.args) >= 14:
                out_multiplier = node.args[12]
                out_shift = node.args[13]
            # If not, compute them.
            else:
                # pyre-ignore[58]: Division operands
                requantize_scale = bias_scale / out_scale
                (out_multiplier, out_shift) = quantize_tensor_multiplier(
                    requantize_scale
                )
            linear_args = (
                im2row,
                linear_weight,
                bias,
                in_zero_point,
                weight_zero_point,
                out_multiplier,
                out_shift,
                out_zero_point,
                None,
            )
        else:
            linear_args = (im2row, linear_weight, bias)

        with graph.inserting_before(node):
            linear_res = graph.call_function(
                self.conv_op_to_linear_op[cast(EdgeOpOverload, node.target)],
                args=linear_args,
            )
            linear_res.meta = node.meta

        # The output of linear is a 3D tensor. However, the output is in NHWC
        # layout by default, because an input vector of size X is multiplied
        # with the weight matrix, i.e., column values are contiguous. If the
        # channel_last is False, we want to transpose this output.
        if not channel_last:
            with graph.inserting_before(node):
                linear_res = graph.call_function(
                    exir_ops.edge.aten.transpose_copy.int,
                    args=(linear_res, 1, 2),
                )
                linear_res.meta = node.meta

        # And finally, we want to view the 3D output of linear op as 4D tensor
        with graph.inserting_before(node):
            out_res = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(linear_res, list(out_shape)),
            )
            out_res.meta = node.meta

        node.replace_all_uses_with(out_res)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceTransposedConvWithLinearPass(RemoveOrReplacePassInterface):
    """
    Replace transposed convolution where groups=1 with transposed_im2row
    followed by a linear op.
    """

    # A map from the transposed_convolution op to the linear op that it should
    # decompose to.
    transposed_conv_op_to_linear_op: Dict[EdgeOpOverload, EdgeOpOverload] = {
        exir_ops.edge.cadence.transposed_convolution.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.quantized_transposed_conv.default: exir_ops.edge.cadence.quantized_linear.default,
    }

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return list(self.transposed_conv_op_to_linear_op.keys())

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the relevant args from transposed_convolution node.
        assert isinstance(node.target, EdgeOpOverload)
        quantized_op = (
            node.target == exir_ops.edge.cadence.quantized_transposed_conv.default
        )
        expected_args = 16 if quantized_op else 9
        if len(node.args) != expected_args:
            return False

        (
            in_tensor,
            weight,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
        ) = node.args[0:8]

        # We do not replace depthwise transposed_convolution with gemm yet.
        if groups != 1:
            return False

        # Get the shapes
        assert isinstance(weight, torch.fx.Node)
        out_shape = node.meta["val"].shape
        weight_shape = weight.meta["val"].shape
        if None in {weight_shape, out_shape}:
            return False

        # Determine if the transposed_convolution is NCHW or NHWC. The NHWC,
        # i.e., the channel_last layout is specified by the channel_last arg
        # of transposed_conv op, which is the last argument.
        channel_last = node.args[-1]
        # The weight tensor is [out_channels, in_channels, X] for NCHW layout,
        # and [out_channels, X, in_channels] for NHWC layout. Here, X is the
        # kernel_width for conv1d, and X = kernel_height * kernel_width for
        # conv2d. We extract X as the kernel_size for im2row.
        kernel_size = list(weight_shape[1:-1] if channel_last else weight_shape[2:])
        assert isinstance(in_tensor, torch.fx.Node)

        # Cast to appropriate types
        stride = cast(Sequence[int], stride)
        padding = cast(Sequence[int], padding)
        dilation = cast(Sequence[int], dilation)
        output_padding = cast(Sequence[int], output_padding)

        # transposed_im2row expects every kernel parameter to be 2d. So we extend the
        # parameters for conv1d by prepending their default values.
        stride_list = ([1] + list(stride)) if len(stride) == 1 else list(stride)
        padding_list = ([0] + list(padding)) if len(padding) == 1 else list(padding)
        dilation_list = ([1] + list(dilation)) if len(dilation) == 1 else list(dilation)
        output_padding_list = (
            ([0] + list(output_padding))
            if len(output_padding) == 1
            else list(output_padding)
        )
        kernel_size = ([1] + kernel_size) if len(kernel_size) == 1 else kernel_size
        # Check that kernel size does not have a 0
        if 0 in kernel_size:
            return False

        graph = node.graph

        # If the transposed_convolution op was quantized, we need the input tensor's
        # zero_point for im2row. Otherwise in_zero_point defaults to a zero tensor.
        # We create the tensor as a graph node using aten.full to avoid
        # DataDependentOutputException during FakeTensor shape propagation.
        in_zero_point_val = node.args[8] if quantized_op else 0
        in_zero_point = graph.call_function(
            exir_ops.edge.aten.full.default,
            args=([1], in_zero_point_val),
            kwargs={"dtype": torch.int32},
        )
        # Insert the node before the current node
        node.prepend(in_zero_point)
        in_zero_point.meta = node.meta

        # Create a transposed_im2row node with the input. This will create a 2d
        # matrix of shape [out_height*out_weight, X*in_channels]. X is as
        # defined in the comment above.
        transposed_im2row_args = (
            in_tensor,
            kernel_size,
            dilation_list,
            padding_list,
            stride_list,
            output_padding_list,
            in_zero_point,
            channel_last,
        )
        with graph.inserting_before(node):
            transposed_im2row = graph.call_function(
                exir_ops.edge.cadence.transposed_im2row.default,
                args=transposed_im2row_args,
            )
            transposed_im2row.meta = node.meta

        # Reshape the weight to [out_channels, in_channels * X]
        K = math.prod(weight_shape[1:])

        # Weight is always a Node, so we need a view_copy operation
        with graph.inserting_before(node):
            linear_weight = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(
                    weight,
                    [weight_shape[0], K],
                ),
            )
            linear_weight.meta = node.meta

        # Create the linear node, which multiplies the 3d input with 2d weight
        # tensors with bias addition. The outermost dimension of the input is
        # the batch size for linear op.
        if quantized_op:
            (
                in_zero_point,
                weight_zero_point,
                bias_scale,
                out_scale,
                out_zero_point,
            ) = node.args[8:13]
            # pyre-ignore[58]: Division operands
            requantize_scale = bias_scale / out_scale
            (out_multiplier, out_shift) = quantize_tensor_multiplier(requantize_scale)
            linear_args = (
                transposed_im2row,
                linear_weight,
                bias,
                in_zero_point,
                weight_zero_point,
                out_multiplier,
                out_shift,
                out_zero_point,
                None,
            )
        else:
            linear_args = (transposed_im2row, linear_weight, bias)

        with graph.inserting_before(node):
            linear_res = graph.call_function(
                self.transposed_conv_op_to_linear_op[cast(EdgeOpOverload, node.target)],
                args=linear_args,
            )
            linear_res.meta = node.meta

        # The output of linear is a 3D tensor. However, the output is in NHWC
        # layout by default, because an input vector of size X is multiplied
        # with the weight matrix, i.e., column values are contiguous. If the
        # channel_last is False, we want to transpose this output.
        if not channel_last:
            with graph.inserting_before(node):
                linear_res = graph.call_function(
                    exir_ops.edge.aten.transpose_copy.int,
                    args=(linear_res, 1, 2),
                )
                linear_res.meta = node.meta

        # And finally, we want to view the 3D output of linear op as 4D tensor
        with graph.inserting_before(node):
            out_res = graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(linear_res, list(out_shape)),
            )
            out_res.meta = node.meta

        node.replace_all_uses_with(out_res)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceNopTransposeOrPermuteWithViewPass(RemoveOrReplacePassInterface):
    """
    If the transpose/permute op does not change the byte order (e.g.,
    transpose/permute from Nx1xHxW to NxHx1xW), then it can be replaced
    by view op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.transpose_copy.int,
            exir_ops.edge.aten.permute_copy.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the input tensor and shape
        in_tensor_node = node.args[0]
        assert isinstance(in_tensor_node, torch.fx.Node)
        in_shape = in_tensor_node.meta["val"].shape
        # Get the output tensor shape
        out_shape = node.meta["val"].shape

        if node.target == exir_ops.edge.aten.transpose_copy.int:
            # Get the two dims to be transposed
            dim0 = cast(int, node.args[1])
            dim1 = cast(int, node.args[2])
            dim0 = dim0 if dim0 >= 0 else len(in_shape) + dim0
            dim1 = dim1 if dim1 >= 0 else len(in_shape) + dim1
            # We can eliminate transpose if (a) the size at dim0 and dim1 is 1;
            # (b) the size at dim0 or dim1 is 1, and dim0 and dim1 are consecutive.
            both_one = in_shape[dim0] == 1 and in_shape[dim1] == 1
            either_one_and_consecutive = abs(dim0 - dim1) == 1 and (
                in_shape[dim0] == 1 or in_shape[dim1] == 1
            )
            if both_one or either_one_and_consecutive:
                with node.graph.inserting_before(node):
                    new_node = node.graph.call_function(
                        exir_ops.edge.aten.view_copy.default,
                        args=(in_tensor_node, list(out_shape)),
                    )
                    new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                return True

        elif node.target == exir_ops.edge.aten.permute_copy.default:
            old_dims = list(range(len(in_shape)))
            new_dims = cast(Sequence[int], node.args[1])
            # If the permute does not change anything, return the input as output.
            if old_dims == list(new_dims):
                node.replace_all_uses_with(in_tensor_node)
                return True
            # Get the old dim order, and the permuted dim order for all dims that
            # are not 1.
            old_order = [
                dim for dim, shape_dim in zip(old_dims, in_shape) if shape_dim != 1
            ]
            new_order = [
                dim for dim, shape_dim in zip(new_dims, out_shape) if shape_dim != 1
            ]
            # If the byte ordering for non-unit dims is unchanged, this is a nop.
            if old_order == new_order:
                with node.graph.inserting_before(node):
                    new_node = node.graph.call_function(
                        exir_ops.edge.aten.view_copy.default,
                        args=(in_tensor_node, list(out_shape)),
                    )
                    new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                return True

        return False

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        result = super().call(graph_module)
        # If this pass made modifications, fuse any cascaded view ops that may have been created
        if result.modified:
            fuse_cascaded_result = FuseCascadedViewOps().call(result.graph_module)

            # True because we are in the 'if modified' block
            return PassResult(fuse_cascaded_result.graph_module, True)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceLinearWithFullyConnectedOpPass(RemoveOrReplacePassInterface):
    """
    If the input of linear/quantized_linear op is a vector, replace it with
    fully_connected op.
    """

    linear_to_fc_op: Dict[EdgeOpOverload, EdgeOpOverload] = {
        # Default variants
        exir_ops.edge.aten.linear.default: exir_ops.edge.cadence.fully_connected.default,
        exir_ops.edge.cadence.quantized_linear.default: exir_ops.edge.cadence.quantized_fully_connected.default,
        # Per-tensor variants
        exir_ops.edge.cadence.quantized_linear.per_tensor: exir_ops.edge.cadence.quantized_fully_connected.per_tensor,
        # Type-specialized variants (int8)
        exir_ops.edge.cadence.quantized_linear_asym8sxasym8s_asym8s.per_tensor: exir_ops.edge.cadence.quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor,
        # Type-specialized variants (uint8)
        exir_ops.edge.cadence.quantized_linear_asym8uxasym8u_asym8u.per_tensor: exir_ops.edge.cadence.quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor,
    }

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return list(self.linear_to_fc_op.keys())

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Extract the input tensor
        in_tensor_arg = node.args[0]
        assert isinstance(in_tensor_arg, torch.fx.Node)
        in_tensor_shape = in_tensor_arg.meta["val"].shape
        leading_dims = math.prod(in_tensor_shape[:-1])
        # If the tensor is not a vector, do nothing.
        if leading_dims != 1:
            return False

        # Replace the linear with fully connected op
        assert isinstance(node.target, EdgeOpOverload)
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                self.linear_to_fc_op[cast(EdgeOpOverload, node.target)],
                args=node.args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


register_cadence_pass(CadencePassAttribute(opt_level=0))(ReplaceScalarWithTensorArgPass)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceScalarTensorWithFullPass(RemoveOrReplacePassInterface):
    """
    aten.scalar_tensor can be replaced by aten.full with a shape of [1].
    scalar_tensor is not supported, so this is an opt_level=0 pass.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            torch.ops.aten.scalar_tensor.default,
            exir_ops.edge.aten.scalar_tensor.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.full.default,
                args=(
                    [1],
                    node.args[0],
                ),
                kwargs={"dtype": torch.float32},
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceFullLikeWithFullPass(RemoveOrReplacePassInterface):
    """
    aten.full_like can be replaced by aten.full with the shape of the arg tensor.
    full_like is not supported, so this is an opt_level=0 pass.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.full_like.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        input_arg = node.args[0]
        assert isinstance(input_arg, torch.fx.Node)
        shape = input_arg.meta["val"].shape
        fill_value = node.args[1]

        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.full.default,
                args=(shape, fill_value),
                kwargs={},
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceInfArgInFullWithValuePass(RemoveOrReplacePassInterface):
    """
    aten.full allows "-inf" and "inf" as inputs. The profiler cannot
    handle that, so replace them with the maximum value of the type.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.full.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:

        new_args = list(node.args)
        fill_value = node.args[1]
        if fill_value == float("-inf"):
            new_args[1] = torch.finfo(torch.float32).min
        elif fill_value == float("inf"):
            new_args[1] = torch.finfo(torch.float32).max
        else:
            return False

        new_args = tuple(new_args)

        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.full.default,
                args=new_args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAtenAvgPoolWithCadenceAvgPoolPass(RemoveOrReplacePassInterface):
    """
    Replace the aten avg_pool op with the cadence custom avg_pool2d op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.avg_pool1d.default,
            exir_ops.edge.aten.avg_pool2d.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Determine if the op is avg_pool1d or avg_pool2d
        avg_pool1d: bool = node.target == exir_ops.edge.aten.avg_pool1d.default

        # Get the input tensor node
        in_tensor_node = node.args[0]
        assert isinstance(in_tensor_node, torch.fx.Node)

        # Replace avg_pool2d with custom avg_pool2d, and if the input tensor is
        # quantized, pass its zero_point tensor as arg to the custom avg_pool2d.
        # stride, padding, ceil_mode, count_include_pad, divisor_override, are
        # the native avg_pool2d args. 'channel_last' denotes NCHW vs NHWC layout,
        # and is False by default.
        kernel_size = node.args[1]
        # When stride is not provided or is empty, PyTorch defaults to kernel_size
        stride = node.args[2] if len(node.args) >= 3 and node.args[2] else kernel_size
        padding = node.args[3] if len(node.args) >= 4 else [0, 0]
        ceil_mode = node.args[4] if len(node.args) >= 5 else False
        count_include_pad = node.args[5] if len(node.args) >= 6 else True
        divisor_override = node.args[6] if len(node.args) >= 7 else None
        zero_point = node.args[7] if len(node.args) >= 8 else None

        graph = node.graph
        out_shape = node.meta["val"].shape

        kernel_size = cast(Sequence[int], kernel_size)
        stride = cast(Sequence[int], stride)
        padding = cast(Sequence[int], padding)

        # If the op is avg_pool1d, then we need to reshape the 3d input to a 4d
        # tensor.
        if avg_pool1d:
            in_shape = list(in_tensor_node.meta["val"].shape)
            assert len(in_shape) == 3, "Expected 3d input for avg_pool1d"
            in_shape_4d = in_shape[:2] + [1] + in_shape[2:]

            with graph.inserting_before(node):
                in_view_node = graph.call_function(
                    exir_ops.edge.aten.view_copy.default,
                    args=(in_tensor_node, in_shape_4d),
                )
                in_view_node.meta = node.meta

            # Extend the kernel_size, stride and padding to 2d
            kernel_size = (
                [1] + list(kernel_size) if len(kernel_size) == 1 else kernel_size
            )
            stride = [1] + list(stride) if len(stride) == 1 else stride
            padding = [0] + list(padding) if len(padding) == 1 else padding

            input_for_pool = in_view_node
        else:
            input_for_pool = in_tensor_node

        # Create a new avg_pool node with the updated args
        new_args = (
            input_for_pool,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            zero_point,
            False,
        )

        with graph.inserting_before(node):
            avg_pool2d_node = graph.call_function(
                exir_ops.edge.cadence.avg_pool2d.default,
                args=new_args,
            )
            avg_pool2d_node.meta = node.meta

        # If the node was avg_pool1d, we again reshape the 4d output to 3d output
        if avg_pool1d:
            with graph.inserting_before(node):
                result_node = graph.call_function(
                    exir_ops.edge.aten.view_copy.default,
                    args=(avg_pool2d_node, list(out_shape)),
                )
                result_node.meta = node.meta
            node.replace_all_uses_with(result_node)
        else:
            node.replace_all_uses_with(avg_pool2d_node)

        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceIm2RowWithViewPass(RemoveOrReplacePassInterface):
    """
    Replace im2row with view when possible (no padding, no dilation, and output spatial dimensions are 1).
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.cadence.im2row.default,
            exir_ops.edge.cadence.im2row.per_tensor,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Check if im2row applies padding. If yes, we cannot replace it with view.
        pad = cast(Sequence[int], node.args[3])
        if any(p != 0 for p in pad):
            return False

        # Check if im2row has dilation. If yes, we cannot replace it with view.
        dilation = cast(Sequence[int], node.args[2])
        if any(d != 1 for d in dilation):
            return False

        # im2row works on 3D or 4D tensors.
        # Output shape[1:-1] will be unit if input spatial dimensions are the same as kernel spatial dimensions.
        output_shape = node.meta["val"].shape
        if math.prod(output_shape[1:-1]) != 1:
            return False

        # Replace im2row with view_copy
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(node.args[0], list(output_shape)),
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceEmptyTensorsWithFullPass(ExportPass):
    """Replaces nodes that produce empty tensors with full nodes."""

    def call_operator(self, op, args, kwargs, meta):
        val = meta.data.get("val", None)
        if isinstance(val, torch.Tensor) and val.numel() == 0:
            return super().call_operator(
                exir_ops.edge.aten.full.default,
                args=(val.shape, 0),
                kwargs={"dtype": val.dtype},
                meta=meta,
            )
        return super().call_operator(op, args, kwargs, meta)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        changed = False
        for module in filter(
            lambda m: isinstance(m, torch.fx.GraphModule), graph_module.modules()
        ):
            module = cast(torch.fx.GraphModule, module)
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                val = node.meta.get("val", None)
                if isinstance(val, torch.Tensor) and val.numel() == 0:
                    with module.graph.inserting_before(node):
                        new_node = module.graph.call_function(
                            exir_ops.edge.aten.full.default,
                            args=(val.shape, 0),
                            kwargs={"dtype": val.dtype},
                        )
                        new_node.meta = node.meta
                    node.replace_all_uses_with(new_node)
                    changed = True

        if changed:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            return super().call(graph_module)

        return PassResult(graph_module, False)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceWhereWithFullArgsWithWhereScalar(RemoveOrReplacePassInterface):
    """Replaces where ops using two full ops as tensors with a scalar
    version.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.where.self]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Check if args[1] and args[2] are full ops
        arg1 = node.args[1]
        arg2 = node.args[2]

        if not isinstance(arg1, torch.fx.Node) or not isinstance(arg2, torch.fx.Node):
            return False

        if (
            arg1.target != exir_ops.edge.aten.full.default
            or arg2.target != exir_ops.edge.aten.full.default
        ):
            return False

        # Get the condition tensor shape
        cond_arg = node.args[0]
        assert isinstance(cond_arg, torch.fx.Node)
        cond_shape = list(cond_arg.meta["val"].shape)

        # Check if the full ops have the same size as the cond tensor
        full1_shape = arg1.args[0]
        full2_shape = arg2.args[0]

        if cond_shape != full1_shape or cond_shape != full2_shape:
            return False

        # Get the scalar values from the full ops
        scalar_value_1 = arg1.args[1]
        scalar_value_2 = arg2.args[1]

        # Replace the where op with a scalar where op
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.cadence.where_Scalar.default,
                args=(cond_arg, scalar_value_1, scalar_value_2),
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


# Adapted from fbcode/pyspeech/opt_passes/replace_ops.py
@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceSplitWithSlicePass(RemoveOrReplacePassInterface):
    """
    split_with_sizes() delegates to slice() op, so perform this replacement here.
    This avoids the expense of delegation from ATen.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.split_with_sizes_copy.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # All the users of this split_with_sizes op must be getitem ops
        if any(user.target != operator.getitem for user in node.users):
            return False

        # Get the slice dim and extent for each split
        slice_ops = self._get_split_sizes(node)
        if slice_ops is None:
            return False

        graph = node.graph

        # Go over each getitem user, and replace it with slice op
        for user in list(node.users.keys()):
            assert user.target == operator.getitem
            item_idx = int(user.args[1])
            assert item_idx < len(slice_ops)
            cur_slice = slice_ops[item_idx]
            with graph.inserting_before(user):
                cur_slice_node = graph.call_function(
                    exir_ops.edge.aten.slice_copy.Tensor,
                    (node.args[0], cur_slice[0], cur_slice[1], cur_slice[2], 1),
                )
                # Metadata copy important
                cur_slice_node.meta = user.meta
            user.replace_all_uses_with(cur_slice_node)

        # Return True to indicate the split node should be removed
        return True

    def _get_split_sizes(self, node: torch.fx.Node) -> Optional[list[tuple[int, ...]]]:
        """For split_with_sizes, return the slice dim and extent for each split."""
        # Parse the args of the split_with_sizes op
        tensor_arg, split_sizes = node.args[0:2]
        assert isinstance(tensor_arg, torch.fx.Node)

        # Get shape from node metadata
        val = tensor_arg.meta.get("val")
        if val is None:
            return None
        in_shape = val.shape

        split_dim = 0 if len(node.args) < 3 else node.args[2]

        # Canonicalize the split dimension
        assert isinstance(split_dim, int)
        split_dim = split_dim if split_dim >= 0 else len(in_shape) + split_dim

        # Create the slice op args corresponding to each split
        slice_ops = []
        split_start = 0
        assert isinstance(split_sizes, list)
        for split_size in split_sizes:
            split_end = split_start + split_size
            slice_args = (split_dim, split_start, split_end)
            slice_ops.append(slice_args)
            split_start = split_end

        return slice_ops


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplacePowWithMulPass(RemoveOrReplacePassInterface):
    """
    Replace the pow op with successive mul ops when the exponent is an
    integer between 2 and 4 (inclusive).
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.pow.Tensor_Scalar]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Check if we have at least 2 args and the exponent is an int
        if len(node.args) < 2 or not isinstance(node.args[1], int):
            return False

        exponent = cast(int, node.args[1])

        # Only replace if exponent is between 2 and 4 (inclusive)
        if exponent < 2 or exponent > 4:
            return False

        x = node.args[0]
        assert isinstance(x, torch.fx.Node)

        graph = node.graph
        result_node = x

        # Create successive mul operations
        # For exponent=2: x * x (1 mul)
        # For exponent=3: (x * x) * x (2 muls)
        # For exponent=4: ((x * x) * x) * x (3 muls)
        for _ in range(exponent - 1):
            with graph.inserting_before(node):
                result_node = graph.call_function(
                    exir_ops.edge.aten.mul.Tensor,
                    args=(result_node, x),
                )
                result_node.meta = node.meta

        node.replace_all_uses_with(result_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceMatmulWithTransposedMatmulPass(RemoveOrReplacePassInterface):
    """
    For certain backends, we have efficient kernels for transposed matmul. We
    replace AxB with AxB' for such backends.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.cadence.quantized_matmul.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # If already transposed, bail
        if len(node.args) >= 9 and node.args[-1] is True:
            return False

        # Get the args
        if len(node.args) == 9:
            (
                X_arg,
                X_zero_point,
                Y_arg,
                Y_zero_point,
                bias,
                out_multiplier,
                out_shift,
                out_zero_point,
                transposed,
            ) = node.args
        elif len(node.args) == 8:
            (
                X_arg,
                X_zero_point,
                Y_arg,
                Y_zero_point,
                bias,
                out_multiplier,
                out_shift,
                out_zero_point,
            ) = node.args
            transposed = False
        else:
            raise AssertionError(
                f"Unexpected number of args for quantized_matmul: {len(node.args)}"
            )

        # If the matmul is already transposed, bail
        if transposed:
            return False

        # Get the second tensor from metadata
        assert isinstance(Y_arg, torch.fx.Node)
        Y_tensor_val = Y_arg.meta.get("val")
        if Y_tensor_val is None:
            return False

        graph = node.graph

        # Create zero bias
        with graph.inserting_before(node):
            zero_bias = graph.call_function(
                exir_ops.edge.aten.full.default,
                args=([Y_tensor_val.size(-1)], 0),
                kwargs={"dtype": torch.int32},
            )
            zero_bias.meta = node.meta

        # Transpose Y_arg
        with graph.inserting_before(node):
            Y_arg_t = graph.call_function(
                exir_ops.edge.aten.transpose_copy.int,
                args=(Y_arg, -1, -2),
            )
            Y_arg_t.meta = node.meta

        # Construct the new args, and create the transposed matmul op
        new_args = (
            X_arg,
            X_zero_point,
            Y_arg_t,
            Y_zero_point,
            zero_bias,
            out_multiplier,
            out_shift,
            out_zero_point,
            True,
        )

        with graph.inserting_before(node):
            new_node = graph.call_function(
                exir_ops.edge.cadence.quantized_matmul.default,
                args=new_args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        result = super().call(graph_module)
        modified = modified or result.modified
        if modified:
            # Fuse any inserted transpose node with transpose/permute nodes
            # surrounding it.
            result = FuseCascadedTransposeOrPermuteOps().call(result.graph_module)
            modified = modified or result.modified
            # Replace permute with transpose.
            result = ReplacePermuteWithTransposePass().call(result.graph_module)
            modified = modified or result.modified

        return PassResult(result.graph_module, modified)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceMulTensorWithMulAndFullOpsPass(RemoveOrReplacePassInterface):
    """
    Extracts a single value argument of mul op to a separate full op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [torch.ops.aten.mul.Tensor]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        x_arg, const_arg = node.args

        # Swap arguments if the order is wrong
        if isinstance(const_arg, torch.fx.Node):
            x_arg, const_arg = const_arg, x_arg

        # Skip if the const_arg is not a scalar
        if not isinstance(const_arg, (float, int)) or not isinstance(
            x_arg, torch.fx.Node
        ):
            return False

        # Cast the const_arg to the dtype of the x_arg
        full_arg = self.resolve_full_arg(x_arg, const_arg)

        full_output_dtype = torch.int32 if isinstance(full_arg, int) else torch.float32

        # Extract an argument to a separate full op.
        with node.graph.inserting_before(node):
            full_node = node.graph.call_function(
                torch.ops.aten.full.default,
                args=([1], full_arg),
                kwargs={"dtype": full_output_dtype},
            )
            full_node.meta = node.meta
            full_node.meta["val"] = [1]
            new_mul_node = node.graph.call_function(
                torch.ops.aten.mul.Tensor, args=(x_arg, full_node)
            )
            new_mul_node.meta = node.meta
        # Replace the old mul with a newly created mul.
        node.replace_all_uses_with(new_mul_node)
        node.graph.erase_node(node)
        return True

    def resolve_full_arg(
        self, x_arg: torch.fx.Node, const_arg: float | int
    ) -> float | int:
        if x_arg.meta["val"].dtype == torch.float32 and isinstance(const_arg, int):
            const_arg = float(const_arg)
        if x_arg.meta["val"].dtype == torch.int32 and isinstance(const_arg, float):
            const_arg = int(const_arg)
        return const_arg


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass(RemoveOrReplacePassInterface):
    """
    Replace the aten adaptive avg_pool op with the aten avg_pool2d op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten._adaptive_avg_pool2d.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the input tensor node
        in_tensor_node = node.args[0]
        assert isinstance(in_tensor_node, torch.fx.Node)

        # Get input shape (in NCHW format)
        in_shape = in_tensor_node.meta["val"].shape
        output_size = cast(Sequence[int], node.args[1])
        num_dims = len(output_size)

        # Spatial dimensions are at indices [2:] for NCHW format
        # TODO: If in_tensor_shape is not a multiple of output size,
        # this pass will not work. T224984800
        dim_multiples = [
            (in_shape[i + 2] % output_size[i]) == 0 for i in range(num_dims)
        ]
        if not all(dim_multiples):
            logging.info(
                f"Unable to replace adaptive average pool with average pool. Input tensor shape of {in_shape} is not a multiple of output size: {output_size}"
            )
            return False

        # Compute stride and kernel_size based on spatial dimensions
        stride = [(in_shape[i + 2] // output_size[i]) for i in range(num_dims)]
        kernel_size = [
            in_shape[i + 2] - (output_size[i] - 1) * stride[i] for i in range(num_dims)
        ]
        padding = [0] * num_dims
        ceil_mode = False
        count_include_pad = True
        divisor_override = None

        # Create a new avg_pool2d node with the computed args
        new_args = (
            in_tensor_node,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.avg_pool2d.default,
                args=new_args,
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceTorchQuantizedEmbeddingWithCadenceQuantizedEmbedding(
    RemoveOrReplacePassInterface
):
    """
    Replace torch.ops.quantized_decomposed.embedding_byte.dtype with
    torch.ops.cadence.quantized_embedding_byte
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.quantized_decomposed.embedding_byte.default,
            exir_ops.edge.quantized_decomposed.embedding_byte.dtype,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Replace with cadence.quantized_embedding_byte
        if len(node.args) < 6:
            raise AssertionError(
                f"Expected 6 arguments for embedding_byte, got {len(node.args)}"
            )
        embedding = node.args[0]
        scales = node.args[1]
        weight_zero_points = node.args[2]
        indices = node.args[5]

        if node.target == exir_ops.edge.quantized_decomposed.embedding_byte.dtype:
            dtype = node.kwargs.get("dtype", None)
            if dtype is not None and dtype != torch.float32:
                raise AssertionError(
                    f"Unsupported output dtype for embedding_byte: {dtype}"
                )

        new_args = (embedding, scales, weight_zero_points, indices, False)
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.cadence.quantized_embedding_byte.default,
                args=new_args,
            )
            new_node.meta = node.meta

        node.replace_all_uses_with(new_node)
        return True


class CommonReplacePasses:
    passes = [
        ReplaceScalarWithTensorArgPass,
        ReplaceSqueezeAndUnsqueezeWithViewPass,
        ReplaceSplitWithSlicePass,
        ReplaceSelectWithViewOpPass,
        ReplaceMMWithAddMMPass,
        ReplaceRepeatWithCatPass,
        ReplaceFullLikeWithFullPass,
        ReplaceAtenConvolutionWithCadenceConvolutionPass,
        ReplacePT2QuantWithCadenceQuantPass,
        ReplacePT2DequantWithCadenceDequantPass,
        ReplacePowWithMulPass,
        ReplaceTorchQuantizedEmbeddingWithCadenceQuantizedEmbedding,
    ]


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAtenLinalgSvdWithCadenceLinalgSvdPass(RemoveOrReplacePassInterface):
    """
    Replace aten linalg svd op with cadence custom op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten._linalg_svd.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.cadence.linalg_svd.default,
                args=node.args,
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True


# This class encapsulates all the functions that replace/switch one op in the
# graph with another.
class CadenceReplaceOpsInGraph:
    passes = CommonReplacePasses.passes + [
        ReplaceAtenLinalgSvdWithCadenceLinalgSvdPass,
        ReplaceEmptyTensorsWithFullPass,
        ReplaceFunctionallyEquivalentOpTargets,
        ReplacePermuteWithTransposePass,
        ReplaceConvolutionOptionalArgsWithConcreteArgsPass,
        ReplaceAddMMWithLinearPass,
        RemoveNopSelectOpPass,
        ReplacePadWithCatPass,
        ReplaceConstantPadNdWithSlicePass,
        ReplaceConvWithChannelLastConvPass,
        ReplaceTrivialConvWithLinear,
        ReplaceConvWithIm2RowAndLinear,
        ReplaceTransposedConvWithLinearPass,
        # This pass should be after passes that replace conv -> im2row + linear.
        ReplaceIm2RowWithViewPass,
        MakeSliceAndCatDimOutermostPass,
        ReplaceMatmulWithTransposedMatmulPass,
        ReplaceNopTransposeOrPermuteWithViewPass,
        ReplaceLinearWithFullyConnectedOpPass,
        ReplaceScalarTensorWithFullPass,
        ReplaceInfArgInFullWithValuePass,
        ReplaceLogicalNotBooleanWhereWithWherePass,
        ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass,
        ReplaceAtenAvgPoolWithCadenceAvgPoolPass,
        ReplaceWhereWithFullArgsWithWhereScalar,
        ReplaceMulTensorWithMulAndFullOpsPass,
    ]
