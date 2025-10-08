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
from typing import cast, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.fx
from executorch.backends.cadence.aot.compiler_utils import (
    get_shape,
    get_tensor_from_attr,
    get_zero_point,
    is_node_with_op,
    quantize_tensor_multiplier,
)
from executorch.backends.cadence.aot.fuse_ops import (
    FuseCascadedTransposeOrPermuteOps,
    FuseCascadedViewOps,
)
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    none_throws,
    register_cadence_pass,
)
from executorch.backends.cadence.aot.remove_ops import RemoveNopSelectOpPass
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import ExportPass, NodeMetadata, PassResult, ProxyValue
from torch.fx.node import Argument

# A map to represent ops that:
# (a) are functionally equivalent; and
# (b) have identical arguments
# An op whose target is 'key' in this dict can be replaced by the functionally euivalent
# op whose target is 'value'. The replacement would just involve changing the op target.
functionally_equivalent_op_targets: Dict[EdgeOpOverload, EdgeOpOverload] = {
    exir_ops.edge.aten.relu_.default: exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.unsafe_split.Tensor: exir_ops.edge.aten.split_copy.Tensor,
}


def contains_placeholder_or_param(nodes: Iterable[torch.fx.Node]) -> bool:
    """
    Return true if any of the node in the incoming nodes list is a placeholder
    or parameter
    """
    return any(
        is_node_with_op(node, "placeholder") or is_node_with_op(node, "get_attr")
        for node in nodes
    )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceLogicalNotBooleanWhereWithWherePass(ExportPass):
    """
    A where op with a logical_not and a boolean tensor can be replaced
    by a where op with flipped inputs and the initial boolean tensor.
    """

    def replace_logical_nop_where_with_where(
        self, graph_module: torch.fx.GraphModule
    ) -> None:
        graph = graph_module.graph
        for node in graph.nodes:
            # We are only interested in where nodes
            if node.target != exir_ops.edge.aten.where.self:
                continue

            # If the third arg is not a logical_not, bail.
            if node.args[0].target != exir_ops.edge.aten.logical_not.default:
                continue

            # Get the third arg node and its input
            logical_not_node = node.args[0]
            logical_not_input_node = logical_not_node.args[0]

            # If the logical_not input is not a boolean tensor, bail.
            if logical_not_input_node.meta["val"].dtype != torch.bool:
                continue

            # Replace the where op with another one, flipping the inputs and using the boolean
            # tensor from logical_not.
            with graph.inserting_before(node):
                linear_node = graph.call_function(
                    exir_ops.edge.aten.where.self,
                    args=(logical_not_node.args[0], node.args[2], node.args[1]),
                )
            # Replace all the uses
            node.replace_all_uses_with(linear_node)

        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.replace_logical_nop_where_with_where(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceSafeSoftmaxWithSoftmax(ExportPass):  # keep
    """
    Replace _safe_softmax with _softmax
    """

    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != torch.ops.aten._safe_softmax.default:
            return super().call_operator(op, args, kwargs, meta)

        # Add False for the half_to_float argument of softmax
        softmax_args = list(args) + [False]

        return super().call_operator(
            torch.ops.aten._softmax.default,
            tuple(softmax_args),
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplacePT2QuantWithCadenceQuantPass(ExportPass):
    """
    Replace the pt2 quantization ops with cadence quantization ops.
    We do not link kernels to the PT2 quantization ops, so we need to
    replace them with cadence ops at all optimization levels.
    """

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        ns = exir_ops.edge if isinstance(op, EdgeOpOverload) else torch.ops
        if op != ns.quantized_decomposed.quantize_per_tensor.default:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            ns.cadence.quantize_per_tensor.default,
            args,
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplacePT2DequantWithCadenceDequantPass(ExportPass):
    """
    Replace the pt2 dequantization ops with cadence dequantization ops.
    We do not link kernels to the PT2 quantization ops, so we need to
    replace them with cadence ops at all optimization levels.
    """

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        ns = exir_ops.edge if isinstance(op, EdgeOpOverload) else torch.ops
        if op != ns.quantized_decomposed.dequantize_per_tensor.default:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            ns.cadence.dequantize_per_tensor.default,
            args,
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceSqueezeAndUnsqueezeWithViewPass(ExportPass):
    """
    When the shape is static, replace squeeze_copy and unsqueeze_copy ops with
    view_copy op
    """

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # Instead of testing EdgeOpOverload, test EdgeOpOverloadPacket,
        # which allows us to cover all overloads.
        if get_edge_overload_packet(op) not in {
            exir_ops.edge.aten.squeeze_copy,
            exir_ops.edge.aten.unsqueeze_copy,
        }:
            return super().call_operator(op, args, kwargs, meta)
        # Get the output tensor shape
        out_shape = meta["val"].shape

        # Bail out if any dim is not an int (dynamic shape)
        for dim in list(out_shape):
            if not isinstance(dim, int):
                return super().call_operator(op, args, kwargs, meta)

        # Return a view op with the new shape
        view_args = (args[0], list(out_shape))
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default, view_args, kwargs, meta
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceFunctionallyEquivalentOpTargets(ExportPass):
    """
    Replace an op with a functionally equivalent op by just switching the op
    target, but without incurring any change to the op args.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in functionally_equivalent_op_targets:
            return super().call_operator(op, args, kwargs, meta)
        return super().call_operator(
            functionally_equivalent_op_targets[op], args, kwargs, meta
        )


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceSelectWithViewOpPass(ExportPass):
    """
    If the size along the select dim is 1, then the select op can be replaced
    by view op.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.select_copy.int:
            return super().call_operator(op, args, kwargs, meta)

        # Glean the shape of input and output tensor
        in_tensor = args[0].to_tensor()
        in_shape = in_tensor.shape
        out_shape = meta["val"].shape
        # Get the select dimension
        select_dim = args[1] if args[1] >= 0 else args[1] + len(in_shape)

        if in_shape[select_dim] == 1:
            # Return a view op with the new shape
            view_args = (args[0], list(out_shape))
            return super().call_operator(
                exir_ops.edge.aten.view_copy.default, view_args, kwargs, meta
            )
        return super().call_operator(op, args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceMMWithAddMMPass(ExportPass):
    """
    This pass replaces mm with addmm by introducing a zero bias.
    mm is not supported, so this is an opt_level=0 pass.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.mm.default:
            return super().call_operator(op, args, kwargs, meta)

        # The mm op has two args: input, mat2
        assert len(args) == 2
        X, mat2 = args

        # Create a zero bias tensor, and insert it as a graph buffer before the
        # current node
        mat2_tensor = mat2.to_tensor()
        bias_size = mat2_tensor.size(1)
        zero_bias = super().call_operator(
            exir_ops.edge.aten.full.default,
            ([bias_size], 0.0),
            {"dtype": torch.float32},
            meta,
        )

        # Replace mm with addmm
        new_args = (zero_bias, X, mat2)
        return super().call_operator(
            exir_ops.edge.aten.addmm.default, new_args, kwargs, meta
        )


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceAddMMWithLinearPass(ExportPass):
    """
    This pass replaces addmm with linear op.
    """

    def __init__(self):
        super().__init__()
        self.counter = 0

    def replace_addmm_with_linear(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            # We are only interested in admm nodes
            if node.target != exir_ops.edge.aten.addmm.default:
                continue

            # The addmm op has three concrete args: input, mat1, mat2
            assert len(node.args) >= 3
            (bias, mat1, mat2) = node.args[0:3]
            # The other two args are optional scale args
            beta = node.kwargs.get("beta", 1.0)
            alpha = node.kwargs.get("alpha", 1.0)

            # AddMM performs beta*bias + alpha*mm(mat1, mat2). We can convert
            # it to linear op by multiplying beta to bias, and alpha to mat2.t().
            # However, the following two conditions must hold:
            # a. If bias is not a param, then beta must be 1.0
            # b. If mat2 is not a param, then mat2 must be a transpose op. Also,
            # the input to the transpose must be a param, or alpha must be 1.0.
            fit_bias = is_node_with_op(bias, "get_attr") or beta == 1.0
            fit_mat2 = is_node_with_op(mat2, "get_attr")
            transposed_mat2 = False
            if (
                not fit_mat2
                and is_node_with_op(mat2, "call_function")
                and mat2.target == exir_ops.edge.aten.transpose_copy.int
            ):
                mat2, transposed_mat2 = mat2.args[0], True
                fit_mat2 = is_node_with_op(mat2, "get_attr") or alpha == 1.0

            if not fit_bias or not fit_mat2:
                continue

            # Multiply bias by beta
            if beta != 1.0:
                assert is_node_with_op(bias, "get_attr")
                bias_tensor = get_tensor_from_attr(graph_module, bias)
                assert isinstance(bias_tensor, torch.Tensor)
                bias_tensor = beta * bias_tensor
                with graph.inserting_before(node):
                    bias_name = f"_bias_addmm_to_linear_{self.counter}"
                    graph_module.register_buffer(bias_name, bias_tensor)
                    bias = graph.get_attr(bias_name)

            # Use associativity of scalar multiplication, and multiply alpha to mat2
            if is_node_with_op(mat2, "get_attr"):
                mat2_tensor = get_tensor_from_attr(graph_module, mat2)
                assert isinstance(mat2_tensor, torch.Tensor)
                mat2_tensor = alpha * mat2_tensor
                # transpose mat2
                mat2_tensor = mat2_tensor if transposed_mat2 else mat2_tensor.t()
                with graph.inserting_before(node):
                    mat2_name = f"_mat2_addmm_to_linear_{self.counter}"
                    graph_module.register_buffer(mat2_name, mat2_tensor)
                    mat2 = graph.get_attr(mat2_name)

            # Construct the linear node
            linear_args = (mat1, mat2, bias)
            with graph.inserting_before(node):
                linear_node = graph.call_function(
                    exir_ops.edge.aten.linear.default, args=linear_args
                )
                linear_node.meta = node.meta
            # Replace all the uses of the addmm op with linear op
            node.replace_all_uses_with(linear_node)
            self.counter += 1

        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.replace_addmm_with_linear(graph_module)
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplacePermuteWithTransposePass(ExportPass):
    """
    Replace permute op with transpose if the permutation is only along
    two dimensions.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.permute_copy.default:
            return super().call_operator(op, args, kwargs, meta)

        # Get the old dim and new dim order
        in_tensor = args[0].to_tensor()
        old_dims = tuple(range(in_tensor.dim()))
        new_dims = args[1]

        # Compute the number of positions in which the old and new order differ
        diff = [od for od, nd in zip(old_dims, new_dims) if od != nd]

        # If the difference is in two dimensions, we can replace this permute op
        # with transpose op.
        if len(diff) == 2:
            new_args = (args[0], diff[0], diff[1])
            return super().call_operator(
                exir_ops.edge.aten.transpose_copy.int, new_args, kwargs, meta
            )

        return (
            args[0] if len(diff) == 0 else super().call_operator(op, args, kwargs, meta)
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceConvolutionOptionalArgsWithConcreteArgsPass(ExportPass):
    """
    Replace optional tensors with concrete tensors. Currently, we
    replace the optional bias tensor with a zero tensor.
    """

    def call_operator(self, op, args, kwargs, meta):
        op_packet = get_edge_overload_packet(op)
        if op_packet not in {
            exir_ops.edge.cadence.convolution,
            exir_ops.edge.cadence.transposed_convolution,
        }:
            return super().call_operator(op, args, kwargs, meta)

        is_transposed = op_packet == exir_ops.edge.cadence.transposed_convolution
        expected_args = 9 if is_transposed else 8
        assert len(args) == expected_args
        # Check if the bias is already concrete
        if args[2] is not None:
            return super().call_operator(op, args, kwargs, meta)

        # The bias length is the number of out channels.
        out_shape = meta["val"].shape
        bias_size = out_shape[1]
        # Create a zero bias tensor (bias is not a constant tensor,
        # so it needs to be the result of a graph operation).
        zero_bias = super().call_operator(
            exir_ops.edge.aten.full.default,
            ([bias_size], 0.0),
            {"dtype": torch.float32},
            meta,
        )

        # Replace bias with zero_bias
        args = list(args)
        args[2] = zero_bias
        args = tuple(args)

        return super().call_operator(op, args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceRepeatWithCatPass(ExportPass):
    """
    Replace repeat op as successive cat ops along different dimensions.
    repeat is not supported, so this is an opt_level=0 pass.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.repeat.default:
            return super().call_operator(op, args, kwargs, meta)

        # Extract the input tensor, and the repeats from the args
        in_tensor = args[0]
        repeats = args[1]

        # Glean the shapes of input tensor
        in_shape = list(in_tensor.to_tensor().shape)

        # If the size of repeats is more than the dimensionality of the tensor,
        # the output of repeat will be a higher-dimensional tensor. We reshape
        # the input so that it has the same dimensionality as the output tensor.
        diff = len(repeats) - len(in_shape)
        assert (
            diff >= 0
        ), "Repeat arg malformed: expected a repeat along each dimension of input tensor"

        if diff > 0:
            # Extend the input shape with 1's along the higher dimensions
            in_shape = ([1] * diff) + in_shape
            # Insert a view op that reshapes the input tensor to have same
            # dimensionality as the output tensor.
            in_tensor = super().call_operator(
                exir_ops.edge.aten.view_copy.default,
                (in_tensor, in_shape),
                kwargs,
                meta,
            )
            assert len(repeats) == len(in_shape)

        # Repeat op is nothing but successive cat ops along each dimension.
        for dim, repeat in reversed(list(enumerate(repeats))):
            # We do not need to do anything if repeat factor is 1
            if repeat == 1:
                continue
            cat_arg = [in_tensor] * repeat
            in_tensor = super().call_operator(
                exir_ops.edge.aten.cat.default, (cat_arg, dim), kwargs, meta
            )

        return in_tensor


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplacePadWithCatPass(ExportPass):
    """
    Replace constant pad nd op that does padding on outer-most dimension
    with Cat(left_padding_constant_tensor, X, right_padding_constant_tensor)
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.constant_pad_nd.default:
            return super().call_operator(op, args, kwargs, meta)

        assert len(args) >= 2
        input_node, orig_padding = args[:2]

        # if there is no padding, this op will be treated in removal pass.
        if not orig_padding:
            return super().call_operator(op, args, kwargs, meta)

        value = 0 if len(args) == 2 else args[2]

        arg_shape = input_node.to_tensor().shape

        padding = orig_padding + ([0] * (len(orig_padding) % 2 != 0))
        assert len(padding) >= 2
        (left_padding_size, right_padding_size) = padding[-2:]
        # Replace only if constant_pad_nd is along the innermost padding dimension.
        if (
            any(x != 0 for x in padding[0:-2])
            or left_padding_size < 0
            or right_padding_size < 0
        ):
            return super().call_operator(op, args, kwargs, meta)

        cat_tensors = []
        dim = len(arg_shape) - len(padding) // 2
        # add left_padding
        if left_padding_size > 0:
            left_padding_shape = (
                arg_shape[:dim] + (left_padding_size,) + arg_shape[dim + 1 :]
            )
            left_padding_node = super().call_operator(
                exir_ops.edge.aten.full.default,
                (
                    left_padding_shape,
                    value,
                ),
                {"dtype": torch.float32},
                meta,
            )
            cat_tensors.append(left_padding_node)
        # input_node
        cat_tensors.append(input_node)
        # right_padding
        if right_padding_size > 0:
            right_padding_shape = (
                arg_shape[:dim] + (right_padding_size,) + arg_shape[dim + 1 :]
            )
            right_padding_node = super().call_operator(
                exir_ops.edge.aten.full.default,
                (
                    right_padding_shape,
                    value,
                ),
                {"dtype": torch.float32},
                meta,
            )
            cat_tensors.append(right_padding_node)

        assert len(cat_tensors) == 1 + (left_padding_size > 0) + (
            right_padding_size > 0
        )

        new_args = (cat_tensors, dim)
        return super().call_operator(
            exir_ops.edge.aten.cat.default,
            new_args,
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceConstantPadNdWithSlicePass(ExportPass):
    """
    Replace constant pad nd op that does padding on outer-most dimension
    with exir_ops slice(left_padding_constant_tensor, X, right_padding_constant_tensor)
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.constant_pad_nd.default:
            return super().call_operator(op, args, kwargs, meta)

        assert len(args) >= 2
        input_node, orig_padding = args[:2]

        # if there is no padding, this op will be treated in removal pass.
        if not orig_padding:
            return super().call_operator(op, args, kwargs, meta)

        padding = orig_padding + ([0] * (len(orig_padding) % 2 != 0))
        assert len(padding) >= 2
        (start, diff) = map(neg, padding[-2:])
        # Replace only if constant_pad_nd is along the innermost padding dimension.
        if any(x != 0 for x in padding[0:-2]) or start < 0 or diff < 0:
            return super().call_operator(op, args, kwargs, meta)

        arg_shape = input_node.to_tensor().shape
        dim = len(arg_shape) - len(padding) // 2
        stop = arg_shape[dim] - diff
        assert start <= stop
        new_args = (input_node, dim, start, stop)
        return super().call_operator(
            exir_ops.edge.aten.slice.Tensor,
            new_args,
            kwargs,
            meta,
        )


# Make that pass runnable standalone at opt level 0.
@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAtenConvolutionWithCadenceConvolutionPass(ExportPass):
    """
    Replace aten convolution op with jarvis-specific convolution op, since the
    aten version is not supported by jarvis.
    Also remove convolution stride if the output size along the strided dimension
    is 1. We can enable more transformations (e.g., conv -> linear replacement)
    for unit-stride convolutions.
    """

    def call_operator(self, op, args, kwargs, meta):
        if get_edge_overload_packet(op) != exir_ops.edge.aten.convolution:
            return super().call_operator(op, args, kwargs, meta)
        # There must be 9 total args.
        assert len(args) == 9

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
        ) = args
        # Currently we only handle conversion to conv1d and conv2d, therefore
        # verify that the stride, padding, dilation, and output_padding have
        # len <=2.
        assert (
            len(stride) == len(padding) == len(dilation) == len(output_padding) == 1
        ) or (
            len(stride) == len(padding) == len(dilation) == len(output_padding) == 2
        ), "Can only map convolution to conv1d and conv2d at present"

        target = (
            exir_ops.edge.cadence.transposed_convolution.default
            if transposed
            else exir_ops.edge.cadence.convolution.default
        )

        if transposed:
            # Flip the height and width dimensions of weight, since we apply a
            # gather stencil. Also, the first two dimensions of weight must be
            # transposed/interchanged.
            # If weight is a ProxyValue, new_weight needs to be the output of a
            # graph operation (in this case a transpose_copy op) to be an explicit
            # ProxyValue as well. If not, the view op can be done directly on the
            # tensor.
            transposed_weight = super().call_operator(
                exir_ops.edge.aten.transpose_copy.int,
                (
                    weight,
                    0,
                    1,
                ),
                kwargs,
                meta,
            )

            flipped_weight = super().call_operator(
                exir_ops.edge.aten.flip.default,
                (
                    transposed_weight,
                    [-1] if transposed_weight.to_tensor().dim() == 3 else [-1, -2],
                ),
                kwargs,
                meta,
            )

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
            assert all(
                x == 0 for x in output_padding
            ), f"Cannot handle padded output in convolution. Got {output_padding=}"

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
                False,
            )

        return super().call_operator(target, new_args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceTrivialConvWithLinear(ExportPass):
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
        exir_ops.edge.cadence.convolution.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.quantized_conv2d_nchw.default: exir_ops.edge.cadence.quantized_linear.default,
        exir_ops.edge.cadence.quantized_conv2d_nhwc.default: exir_ops.edge.cadence.quantized_linear.default,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.trivial_conv_op_to_linear_op:
            return super().call_operator(op, args, kwargs, meta)

        # Parse the necessary args of the convolution node. Both convolution
        # and quantized_conv have the same first 8 args. The quantized op has
        # extra args holding at least the zero point and scale of input, weight, bias,
        # and output tensor.
        quantized_op = (
            op == exir_ops.edge.cadence.quantized_conv2d_nchw.default
            or op == exir_ops.edge.cadence.quantized_conv2d_nhwc.default
        )
        assert (len(args) == 8 and not quantized_op) or (
            len(args) >= 12 and quantized_op
        ), "Inconsistent args for convolution"
        (in_tensor, weight, bias, stride, padding, dilation, groups) = args[0:7]

        # Glean the shapes of input, weight, and output
        in_shape = in_tensor.to_tensor().shape

        weight_shape = weight.to_tensor().shape
        out_shape = meta["val"].shape
        assert None not in {in_shape, weight_shape, out_shape}

        # Check the condition under which conv can be replaced by linear: (1) this
        # should not be a depthwise convolution; (2) the padding, stride, and dilation
        # should be standard; (3) The [channels, height, width] of input must match the
        # [channel, kernel_height, kernel_width] of the weight. These conditions would
        # ensure that output height and width are 1, and the convolution can be replaced
        # by linear.
        if (
            groups != 1
            or any(x != 0 for x in padding)
            or any(x != 1 for x in stride)
            or any(x != 1 for x in dilation)
            or (list(in_shape[1:]) != list(weight_shape[1:]))
        ):
            return super().call_operator(op, args, kwargs, meta)

        # Reshape the weight to [out_channels, in_channels * X]
        K = math.prod(weight_shape[1:])

        # Weight is always a ProxyValue, so we need a view_copy operation
        linear_weight = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (
                weight,
                [weight_shape[0], K],
            ),
            kwargs,
            meta,
        )

        # Reshape the input from 3d to 2d tensor
        in_view = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (
                in_tensor,
                [in_shape[0], K],
            ),
            kwargs,
            meta,
        )
        # Create the linear node, which multiplies the 2d input and weight
        # tensors, and adds the 1d bias to produce a 2d output.
        if quantized_op:
            (
                in_zero_point,
                weight_zero_point,
                bias_scale,
                out_scale,
                out_zero_point,
            ) = args[7:12]
            # If the multiplier and shift tensors are provided, use them.
            if len(args) >= 14:
                out_multiplier = args[12]
                out_shift = args[13]
            # If not, compute them.
            else:
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

        linear_res = super().call_operator(
            self.trivial_conv_op_to_linear_op[op],
            linear_args,
            kwargs,
            meta,
        )
        # Reshape the output of linear from 2d to 3d tensor
        out_res = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (linear_res, list(out_shape)),
            kwargs,
            meta,
        )
        return out_res


def canonicalize_transposed_dim(dim: int, shape: Sequence[int]) -> int:
    """Canonicalize transpose ops so it gets easier to pattern-match and fuse transpose ops."""
    if dim < 0:
        # Keep transpose dimensions positive.
        dim += len(shape)
    return dim


class ExportPassWithTransposeHelper(ExportPass):
    def transpose_dims(
        self: ExportPass, proxy: ProxyValue, meta: NodeMetadata, dim0: int, dim1: int
    ) -> ProxyValue:
        """Helper function to transpose dims of a `proxy` with given `meta`."""
        shape = proxy.data.shape
        dim0, dim1 = (
            canonicalize_transposed_dim(dim0, shape),
            canonicalize_transposed_dim(dim1, shape),
        )
        dim0, dim1 = min(dim0, dim1), max(dim0, dim1)
        return super().call_operator(
            exir_ops.edge.aten.transpose_copy.int, (proxy, dim0, dim1), {}, meta
        )


@register_cadence_pass(CadencePassAttribute(opt_level=3))
class ReplaceConvWithChannelLastConvPass(ExportPassWithTransposeHelper):
    def change_nchw_to_nhwc(self, proxy: ProxyValue, meta: NodeMetadata) -> ProxyValue:
        shape = proxy.to_tensor().shape
        if len(shape) == 3:
            return self.transpose_dims(proxy, meta, 1, -1)
        indices = list(range(len(shape)))
        permute_indices = [indices[0]] + indices[2:] + [indices[1]]
        return super().call_operator(
            exir_ops.edge.aten.permute_copy.default, (proxy, permute_indices), {}, meta
        )

    def change_nhwc_to_nchw(self, proxy: ProxyValue, meta: NodeMetadata) -> ProxyValue:
        shape = proxy.to_tensor().shape
        if len(shape) == 3:
            return self.transpose_dims(proxy, meta, 1, -1)
        indices = list(range(len(shape)))
        permute_indices = [indices[0], indices[-1]] + indices[1:-1]
        return super().call_operator(
            exir_ops.edge.aten.permute_copy.default, (proxy, permute_indices), {}, meta
        )

    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.cadence.convolution.default,
            exir_ops.edge.cadence.quantized_conv2d_nchw.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        quantized_op = op == exir_ops.edge.cadence.quantized_conv2d_nchw.default

        if not quantized_op and len(args) == 8 and args[-1] is True:
            # Already in NHWC layout.
            return super().call_operator(op, args, kwargs, meta)

        new_op = (
            exir_ops.edge.cadence.quantized_conv2d_nhwc.default
            if quantized_op
            else exir_ops.edge.cadence.convolution.default
        )

        input_proxy = cast(ProxyValue, args[0])
        weight_proxy = cast(ProxyValue, args[1])
        input_proxy = self.change_nchw_to_nhwc(input_proxy, meta)
        weight_proxy = self.change_nchw_to_nhwc(weight_proxy, meta)

        # Non-quantized ops still need to set the last optional argument to True.
        channel_last_arg = [] if quantized_op else [True]

        new_args = (
            # Transposed input/weights.
            (input_proxy, weight_proxy)
            # All other args (bias, quant params, etc)
            + tuple(args[2:])
            + tuple(channel_last_arg)
        )
        output_proxy = super().call_operator(new_op, new_args, kwargs, meta)
        nchw_proxy = self.change_nhwc_to_nchw(output_proxy, meta)
        return nchw_proxy


@register_cadence_pass(CadencePassAttribute(opt_level=3))
class MakeSliceAndCatDimOutermostPass(ExportPassWithTransposeHelper):
    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.slice_copy.Tensor,
        }:
            return super().call_operator(op, args, kwargs, meta)
        dim = cast(int, args[1]) if len(args) > 1 else 0
        output_shape = meta["val"].shape
        if dim < 0:
            # Keep dim positive.
            dim += len(output_shape)

        if dim == 0 or math.prod(output_shape[:dim]) == 1:
            # Not needed if dim is already outermost or all dims before it are 1.
            return super().call_operator(op, (args[0], dim) + args[2:], kwargs, meta)

        if op == exir_ops.edge.aten.slice_copy.Tensor:
            # Transpose -> slice.
            slice_args = (
                self.transpose_dims(cast(ProxyValue, args[0]), meta, dim, 0),
                0,
            ) + args[2:]
            new_op = super().call_operator(op, slice_args, kwargs, meta)
        else:
            # (Transpose input0, Transpose input1, ...) -> cat.
            cat_in_tensors = [
                self.transpose_dims(t, meta, dim, 0)
                for t in cast(list[ProxyValue], args[0])
            ]
            new_op = super().call_operator(op, (cat_in_tensors, 0), kwargs, meta)
        # slice/cat -> transpose.
        return self.transpose_dims(new_op, meta, 0, dim)


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceConvWithIm2RowAndLinear(ExportPass):
    """
    Replace convolution where groups=1 with im2row followed by a linear op.
    """

    # A map from the convolution op to the linear op that it should
    # decompose to.
    conv_op_to_linear_op: Dict[EdgeOpOverload, EdgeOpOverload] = {
        exir_ops.edge.cadence.convolution.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.quantized_conv2d_nchw.default: exir_ops.edge.cadence.quantized_linear.default,
        exir_ops.edge.cadence.quantized_conv2d_nhwc.default: exir_ops.edge.cadence.quantized_linear.default,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.conv_op_to_linear_op:
            return super().call_operator(op, args, kwargs, meta)

        # Get the relevant args from convolution node.
        quantized_op = (
            op == exir_ops.edge.cadence.quantized_conv2d_nchw.default
            or op == exir_ops.edge.cadence.quantized_conv2d_nhwc.default
        )
        assert (len(args) == 8 and not quantized_op) or (
            len(args) >= 12 and quantized_op
        ), "Inconsistent args for convolution"
        (in_tensor, weight, bias, stride, padding, dilation, groups) = args[0:7]

        # We do not replace depthwise convolution with gemm yet.
        if groups != 1:
            return super().call_operator(op, args, kwargs, meta)

        weight_shape = weight.to_tensor().shape
        # If this is a pointwise convolution, im2col will start dominating the
        # runtime. So we call convolution op for this case.
        if (
            all(x == 1 for x in weight_shape[2:])
            and all(x == 1 for x in stride)
            and all(x == 0 for x in padding)
            and all(x == 1 for x in dilation)
        ):
            return super().call_operator(op, args, kwargs, meta)

        # Get the shapes
        out_shape = meta["val"].shape
        assert None not in {weight_shape, out_shape}

        # Determine if the convolution is NCHW or NHWC. The NHWC, i.e., the
        # channel_last layout is specified by the channel_last arg of conv
        # op, which is either the last argument (15th) or implicitely False
        # if the op is quantized, or the last argument if not.
        channel_last = op == exir_ops.edge.cadence.quantized_conv2d_nhwc.default
        # The weight tensor is [out_channels, in_channels, X] for NCHW layout,
        # and [out_channels, X, in_channels] for NHWC layout. Here, X is the
        # kernel_width for conv1d, and X = kernel_height * kernel_width for
        # conv2d. We extract X as the kernel_size for im2row.
        kernel_size = list(weight_shape[1:-1] if channel_last else weight_shape[2:])
        # If the convolution op was quantized, we need the input tensor's
        # zero_point for im2row. Otherwise in_zero_point defaults to a zero
        # tensor.
        in_zero_point = (
            (
                super().call_operator(
                    exir_ops.edge.aten.full.default,
                    (
                        [1],
                        args[7],
                    ),
                    {"dtype": torch.int32},
                    meta,
                )
            )
            if quantized_op
            else torch.tensor(0, dtype=torch.int32)
        )
        # im2row expects every kernel parameter to be 2d. So we extend the
        # parameters for conv1d by prepending their default values.
        stride = ([1] + stride) if len(stride) == 1 else stride
        padding = ([0] + padding) if len(padding) == 1 else padding
        dilation = ([1] + dilation) if len(dilation) == 1 else dilation
        kernel_size = ([1] + kernel_size) if len(kernel_size) == 1 else kernel_size
        # Assert that kernel size does not have a 0
        assert 0 not in kernel_size

        # Create an im2row node with the input. This will create a 2d matrix of
        # shape [out_height*out_weight, X*in_channels]. X is as defined in the
        # comment above.
        im2row_args = (
            in_tensor,
            kernel_size,
            dilation,
            padding,
            stride,
            in_zero_point,
            channel_last,
        )
        im2row = super().call_operator(
            exir_ops.edge.cadence.im2row.default,
            im2row_args,
            kwargs,
            meta,
        )

        # Get the product of the >2 dims of the weight
        K = math.prod(weight_shape[1:])

        # Weight is always a ProxyValue, so we need a view_copy operation
        linear_weight = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (
                weight,
                [weight_shape[0], K],
            ),
            kwargs,
            meta,
        )

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
            ) = args[7:12]
            # If the multiplier and shift tensors are provided, use them.
            if len(args) >= 14:
                out_multiplier = args[12]
                out_shift = args[13]
            # If not, compute them.
            else:
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
        linear_res = super().call_operator(
            self.conv_op_to_linear_op[op],
            linear_args,
            kwargs,
            meta,
        )
        # The output of linear is a 3D tensor. However, the output is in NHWC
        # layout by default, because an input vector of size X is multiplied
        # with the weight matrix, i.e., column values are contiguous. If the
        # channel_last is False, we want to transpose this output.
        if not channel_last:
            linear_res = super().call_operator(
                exir_ops.edge.aten.transpose_copy.int,
                (linear_res, 1, 2),
                kwargs,
                meta,
            )
        # And finally, we want to view the 3D output of linear op as 4D tensor
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (linear_res, list(out_shape)),
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceTransposedConvWithLinearPass(ExportPass):
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

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.transposed_conv_op_to_linear_op:
            return super().call_operator(op, args, kwargs, meta)

        # Get the relevant args from transposed_convolution node.
        quantized_op = op == exir_ops.edge.cadence.quantized_transposed_conv.default
        assert len(args) == (
            16 if quantized_op else 9
        ), "Inconsistent args for transposed_convolution"
        (
            in_tensor,
            weight,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
        ) = args[0:8]

        # We do not replace depthwise transposed_convolution with gemm yet.
        if groups != 1:
            return super().call_operator(op, args, kwargs, meta)

        # Get the shapes
        out_shape = meta["val"].shape
        weight_shape = weight.to_tensor().shape
        assert None not in {weight_shape, out_shape}

        # Determine if the transposed_convolution is NCHW or NHWC. The NHWC,
        # i.e., the channel_last layout is specified by the channel_last arg
        # of transposed_conv op, which is the last argument.
        channel_last = args[-1]
        # The weight tensor is [out_channels, in_channels, X] for NCHW layout,
        # and [out_channels, X, in_channels] for NHWC layout. Here, X is the
        # kernel_width for conv1d, and X = kernel_height * kernel_width for
        # conv2d. We extract X as the kernel_size for im2row.
        kernel_size = list(weight_shape[1:-1] if channel_last else weight_shape[2:])
        # If the transposed_convolution op was quantized, we need the input tensor's
        # zero_point for im2row. Otherwise in_zero_point defaults to a zero
        # tensor.
        in_zero_point = (
            get_zero_point(in_tensor.to_tensor())
            if quantized_op
            else torch.tensor(0, dtype=torch.int32)
        )
        # transposed_im2row expects every kernel parameter to be 2d. So we extend the
        # parameters for conv1d by prepending their default values.
        stride = ([1] + stride) if len(stride) == 1 else stride
        padding = ([0] + padding) if len(padding) == 1 else padding
        dilation = ([1] + dilation) if len(dilation) == 1 else dilation
        output_padding = (
            ([0] + output_padding) if len(output_padding) == 1 else output_padding
        )
        kernel_size = ([1] + kernel_size) if len(kernel_size) == 1 else kernel_size
        # Assert that kernel size does not have a 0
        assert 0 not in kernel_size

        # Create a transposed_im2row node with the input. This will create a 2d
        # matrix of shape [out_height*out_weight, X*in_channels]. X is as
        # defined in the comment above.
        transposed_im2row_args = (
            in_tensor,
            kernel_size,
            dilation,
            padding,
            stride,
            output_padding,
            in_zero_point,
            channel_last,
        )
        transposed_im2row = super().call_operator(
            exir_ops.edge.cadence.transposed_im2row.default,
            transposed_im2row_args,
            kwargs,
            meta,
        )
        # Reshape the weight to [out_channels, in_channels * X]
        K = math.prod(weight_shape[1:])

        # Weight is always a ProxyValue, so we need a view_copy operation
        linear_weight = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (
                weight,
                [weight_shape[0], K],
            ),
            kwargs,
            meta,
        )

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
            ) = args[8:13]
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
        linear_res = super().call_operator(
            self.transposed_conv_op_to_linear_op[op],
            linear_args,
            kwargs,
            meta,
        )
        # The output of linear is a 3D tensor. However, the output is in NHWC
        # layout by default, because an input vector of size X is multiplied
        # with the weight matrix, i.e., column values are contiguous. If the
        # channel_last is False, we want to transpose this output.
        if not channel_last:
            linear_res = super().call_operator(
                exir_ops.edge.aten.transpose_copy.int,
                (linear_res, 1, 2),
                kwargs,
                meta,
            )
        # And finally, we want to view the 3D output of linear op as 4D tensor
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (linear_res, list(out_shape)),
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceNopTransposeOrPermuteWithViewPass(ExportPass):
    """
    If the transpose/permute op does not change the byte order (e.g.,
    transpose/permute from Nx1xHxW to NxHx1xW), then it can be replaced
    by view op.
    """

    def call_operator(self, op, args, kwargs, meta):
        # Only proceed for transpose or permute op.
        if op not in {
            exir_ops.edge.aten.transpose_copy.int,
            exir_ops.edge.aten.permute_copy.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        # Get the input tensor and shape
        in_tensor = args[0].to_tensor()
        in_shape = in_tensor.shape
        # Get the output tensor shape
        out_shape = meta["val"].shape

        if op == exir_ops.edge.aten.transpose_copy.int:
            # Get the two dims to be transposed
            dim0 = args[1] if args[1] >= 0 else in_tensor.dim() + args[1]
            dim1 = args[2] if args[2] >= 0 else in_tensor.dim() + args[2]
            # We can eliminate transpose if (a) the size at dim0 and dim1 is 1;
            # (b) the size at dim0 or dim1 is 1, and dim0 and dim1 are consecutive.
            both_one = in_shape[dim0] == 1 and in_shape[dim1] == 1
            either_one_and_consecutive = abs(dim0 - dim1) == 1 and (
                in_shape[dim0] == 1 or in_shape[dim1] == 1
            )
            if both_one or either_one_and_consecutive:
                new_args = (args[0], list(out_shape))
                return super().call_operator(
                    exir_ops.edge.aten.view_copy.default, new_args, kwargs, meta
                )

        elif op == exir_ops.edge.aten.permute_copy.default:
            old_dims = list(range(in_tensor.dim()))
            new_dims = args[1]
            # If the permute does not change anything, return the input as output.
            if old_dims == new_dims:
                return args[0]
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
                new_args = (args[0], list(out_shape))
                return super().call_operator(
                    exir_ops.edge.aten.view_copy.default, new_args, kwargs, meta
                )

        return super().call_operator(op, args, kwargs, meta)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        result = super().call(graph_module)
        fuse_cascaded_result = none_throws(FuseCascadedViewOps()(result.graph_module))
        result = none_throws(ExportPass()(fuse_cascaded_result.graph_module))
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceLinearWithFullyConnectedOpPass(ExportPass):
    """
    If the input of linear/quantized_linear op is a vector, replace it with
    fully_connected op.
    """

    linear_to_fc_op: Dict[EdgeOpOverload, EdgeOpOverload] = {
        exir_ops.edge.aten.linear.default: exir_ops.edge.cadence.fully_connected.default,
        exir_ops.edge.cadence.quantized_linear.default: exir_ops.edge.cadence.quantized_fully_connected.default,
    }

    def call_operator(self, op, args, kwargs, meta):
        # Only proceed for linear or quantized_linear ops.
        if op not in self.linear_to_fc_op:
            return super().call_operator(op, args, kwargs, meta)

        # Extract the input tensor
        in_tensor = args[0].to_tensor()
        leading_dims = math.prod(in_tensor.shape[:-1])
        # If the tensor is not a vector, do nothing.
        if leading_dims != 1:
            return super().call_operator(op, args, kwargs, meta)

        # Replace the linear with fully connected op
        return super().call_operator(
            self.linear_to_fc_op[op],
            args,
            kwargs,
            meta,
        )


register_cadence_pass(CadencePassAttribute(opt_level=0))(ReplaceScalarWithTensorArgPass)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceScalarTensorWithFullPass(ExportPass):
    """
    aten.scalar_tensor can be replaced by aten.full with a shape of [1].
    scalar_tensor is not supported, so this is an opt_level=0 pass.
    """

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.aten.scalar_tensor.default,
            torch.ops.aten.scalar_tensor.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.aten.full.default,
            (
                [1],
                args[0],
            ),
            {"dtype": torch.float32},
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceFullLikeWithFullPass(ExportPass):
    """
    aten.full_like can be replaced by aten.full with the shape of the arg tensor.
    full_like is not supported, so this is an opt_level=0 pass.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in {
            exir_ops.edge.aten.full_like.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        # Get the shape of the "like" tensor, and pass that in to the full op.
        return super().call_operator(
            exir_ops.edge.aten.full.default,
            (
                args[0].to_tensor().shape,
                args[1],
            ),
            {},
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceInfArgInFullWithValuePass(ExportPass):
    """
    aten.full allows "-inf" and "inf" as inputs. The profiler cannot
    handle that, so replace them with the maximum value of the type.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in {
            exir_ops.edge.aten.full.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        new_args = list(args)

        if args[1] == float("-inf"):
            new_args[1] = torch.finfo(torch.float32).min
        elif args[1] == float("inf"):
            new_args[1] = torch.finfo(torch.float32).max

        return super().call_operator(op, tuple(new_args), kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass(ExportPass):
    """
    Replace ops with single element arguments (size = [1]) with overloads that accept scalar ints/floats.
    """

    # Keep track of which operators and arguments are being replaced.
    replaced_scalar_args: dict[
        EdgeOpOverloadPacket, tuple[EdgeOpOverload, Sequence[int]]
    ] = {
        exir_ops.edge.cadence.quantized_add.default: (
            exir_ops.edge.cadence.quantized_add.per_tensor,
            [1, 2, 4, 5],
        ),
        exir_ops.edge.cadence.quantized_conv2d_nchw.default: (
            exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor,
            [8, 9, 12, 13],
        ),
        exir_ops.edge.cadence.quantized_conv2d_nhwc.default: (
            exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
            [8, 9, 12, 13],
        ),
        exir_ops.edge.cadence.quantized_fully_connected.default: (
            exir_ops.edge.cadence.quantized_fully_connected.per_tensor,
            [4, 5, 6],
        ),
        exir_ops.edge.cadence.quantized_layer_norm.default: (
            exir_ops.edge.cadence.quantized_layer_norm.per_tensor,
            [1, 2],
        ),
        exir_ops.edge.cadence.quantized_linear.default: (
            exir_ops.edge.cadence.quantized_linear.per_tensor,
            [4, 5, 6],
        ),
        exir_ops.edge.cadence.quantized_relu.default: (
            exir_ops.edge.cadence.quantized_relu.per_tensor,
            [1, 3, 4],
        ),
        exir_ops.edge.cadence.im2row.default: (
            exir_ops.edge.cadence.im2row.per_tensor,
            [5],
        ),
        exir_ops.edge.cadence.requantize.default: (
            exir_ops.edge.cadence.requantize.per_tensor,
            [1, 2, 3, 4],
        ),
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.replaced_scalar_args:
            return super().call_operator(op, args, kwargs, meta)

        # Get all the args that need to be replaced.
        new_op, args_to_be_replaced = self.replaced_scalar_args[op]

        if op == new_op:
            return super().call_operator(op, args, kwargs, meta)

        updated_args = list(args)
        for op_arg_index in args_to_be_replaced:
            arg = args[op_arg_index]
            if not isinstance(arg, ProxyValue) or not arg.is_tensor():
                return super().call_operator(op, args, kwargs, meta)

            if not isinstance(arg.node.target, EdgeOpOverload):
                return super().call_operator(op, args, kwargs, meta)

            if get_edge_overload_packet(arg.node.target) != exir_ops.edge.aten.full:
                # Only replace if arg generated by a full op.
                return super().call_operator(op, args, kwargs, meta)

            if tuple(arg.node.args[0]) != (1,):
                # Only replace if the size of the full op is [1].
                return super().call_operator(op, args, kwargs, meta)

            updated_args[op_arg_index] = arg.node.args[1]

        return super().call_operator(
            new_op,
            tuple(updated_args),
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAtenAvgPoolWithCadenceAvgPoolPass(ExportPass):
    """
    Replace the aten avg_pool op with the cadence custom avg_pool2d op.
    """

    def call_operator(self, op, args, kwargs, meta):
        # Only continue for avg_pool op
        if op not in {
            exir_ops.edge.aten.avg_pool1d.default,
            exir_ops.edge.aten.avg_pool2d.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        # Determine if the op is avg_pool1d or avg_pool2d
        avg_pool1d: bool = op == exir_ops.edge.aten.avg_pool1d.default
        # Get the input tensor
        in_tensor = args[0].to_tensor()

        # Replace avg_pool2d with custom avg_pool2d, and if the input tensor is
        # quantized, pass its zero_point tensor as arg to the custom avg_pool2d.
        # stride, padding, ceil_mode, count_include_pad, divisor_override, are
        # the native avg_pool2d args. 'channel_last' denotes NCHW vs NHWC layout,
        # and is False by default.
        kernel_size = args[1]
        stride = args[2] if len(args) >= 3 else [1, 1]
        padding = args[3] if len(args) >= 4 else [0, 0]
        ceil_mode = args[4] if len(args) >= 5 else False
        count_include_pad = args[5] if len(args) >= 6 else True
        divisor_override = args[6] if len(args) >= 7 else None
        zero_point = args[7] if len(args) >= 8 else None

        # If the op is avg_pool1d, then we need to reshape the 3d input to a 4d
        # tensor.
        if avg_pool1d:
            in_shape = list(in_tensor.shape)
            assert len(in_shape) == 3, "Expected 3d input for avg_pool1d"
            in_shape.insert(2, 1)
            out_shape = meta["val"].shape
            in_view_op = super().call_operator(
                exir_ops.edge.aten.view_copy.default,
                (in_tensor, in_shape),
                kwargs,
                meta,
            )
            # Extend the kernel_size, stride and padding to 2d
            kernel_size = [1] + kernel_size if len(kernel_size) == 1 else kernel_size
            stride = [1] + stride if len(stride) == 1 else stride
            padding = [0] + padding if len(padding) == 1 else padding

        # Create a new avg_pool node with the updated args
        new_args = (
            in_view_op if avg_pool1d else args[0],
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            zero_point,
            False,
        )
        avg_pool2d_op = super().call_operator(
            exir_ops.edge.cadence.avg_pool2d.default,
            new_args,
            kwargs,
            meta,
        )

        # If the node was avg_pool1d, we again reshape the 4d output to 3d output
        return (
            super().call_operator(
                exir_ops.edge.aten.view_copy.default,
                (avg_pool2d_op, list(out_shape)),
                kwargs,
                meta,
            )
            if avg_pool1d
            else avg_pool2d_op
        )


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceIm2RowWithViewPass(ExportPass):
    def can_replace(self, op, args, kwargs, meta) -> bool:
        if op != exir_ops.edge.cadence.im2row.default:
            return False

        # Check if im2row applies padding. If yes, we cannot replace it with view.
        pad = cast(tuple[int, ...], args[3])
        if any(p != 0 for p in pad):
            return False

        # Check if im2row has dilation. If yes, we cannot replace it with view.
        dilation = cast(tuple[int, ...], args[2])
        if any(d != 1 for d in dilation):
            return False

        # im2row works on 3D or 4D tensors.
        # Output shape[1:-1] will be unit if input spatial dimensions are the same as kernel spatial dimensions.
        output_shape = meta["val"].shape
        if math.prod(output_shape[1:-1]) == 1:
            return True

        return False

    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.cadence.im2row.default:
            return super().call_operator(op, args, kwargs, meta)

        if not self.can_replace(op, args, kwargs, meta):
            return super().call_operator(op, args, kwargs, meta)

        output_shape = meta["val"].shape
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (args[0], tuple(output_shape)),
            kwargs,
            meta,
        )


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
        ret = super().call(graph_module)
        modified = ret.graph_module.graph.eliminate_dead_code() or ret.modified
        return PassResult(ret.graph_module, modified)


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceWhereWithFullArgsWithWhereScalar(ExportPass):
    """Replaces where ops using two full ops as tensors with a scalar
    version.
    """

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.aten.where.self,
        }:
            return super().call_operator(op, args, kwargs, meta)

        # If the args are not full ops, bail
        # pyre-ignore[16]: `ProxyValue` has no attribute `node`.
        if (args[1].node.target != exir_ops.edge.aten.full.default) or (
            args[2].node.target != exir_ops.edge.aten.full.default
        ):
            return super().call_operator(op, args, kwargs, meta)

        # If one of the full ops is a different size than than the cond tensor, we need to broadcast. Bail.
        if (
            # pyre-ignore[16]: `ProxyValue` has no attribute `node`.
            list(args[0].to_tensor().shape) != args[1].node.args[0]
            or list(args[0].to_tensor().shape) != args[2].node.args[0]
        ):
            return super().call_operator(op, args, kwargs, meta)

        # Get the scalar values from the full ops
        scalar_value_1 = args[1].node.args[1]
        scalar_value_2 = args[2].node.args[1]

        # Replace the where op with a scalar where op
        return super().call_operator(
            exir_ops.edge.cadence.where_Scalar.default,
            (args[0], scalar_value_1, scalar_value_2),
            kwargs,
            meta,
        )

        return super().call_operator(op, args, kwargs, meta)


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAtenApproxGeluWithApproxGeluPass(ExportPass):
    """
    Replace the aten gelu op with an approximate arg with an approximate gelu op.
    """

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in {
            exir_ops.edge.aten.gelu.default,
        }:
            return super().call_operator(op, args, kwargs, meta)
        return super().call_operator(op, args, kwargs, meta)


# Adapted from fbcode/pyspeech/opt_passes/replace_ops.py
@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceSplitWithSlicePass(ExportPass):
    """
    split_with_sizes() delegates to slice() op, so perform this replacement here.
    This avoids the expense of delegation from ATen.
    """

    # For split_with_sizes, return the slice dim and extent for each split.
    def get_split_sizes(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> Optional[list[tuple[int, ...]]]:
        # Parse the args of the split_with_sizes op
        tensor_arg, split_sizes = node.args[0:2]
        assert isinstance(tensor_arg, torch.fx.Node)
        in_shape = get_shape(graph_module, tensor_arg)
        split_dim = 0 if len(node.args) < 3 else node.args[2]
        if in_shape is None:
            return None

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

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if not isinstance(node.target, EdgeOpOverload):
                continue
            if (
                get_edge_overload_packet(node.target)
                != exir_ops.edge.aten.split_with_sizes_copy
            ):
                continue
            # All the users of this split_with_sizes op must be getitem ops
            if any(user.target != operator.getitem for user in node.users):
                continue

            # Get the slice dim and extent for each split
            slice_ops = self.get_split_sizes(graph_module, node)
            if slice_ops is None:
                continue

            # Go over each getitem user, and replace it with slice op
            for user in list(node.users.keys()):
                assert user.target == operator.getitem
                item_idx = user.args[1]
                assert item_idx < len(slice_ops)
                cur_slice = slice_ops[item_idx]
                with graph.inserting_before(user):
                    cur_slice_node = graph.call_function(
                        exir_ops.edge.aten.slice_copy.Tensor,
                        (node.args[0], cur_slice[0], cur_slice[1], cur_slice[2], 1),
                    )
                user.replace_all_uses_with(cur_slice_node)
                graph.erase_node(user)

            graph.erase_node(node)

        graph_module.recompile()
        result = super().call(graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplacePowWithMulPass(ExportPass):
    """
    Replace the pow op for a mul op.
    """

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if not (
            len(args) > 1
            and isinstance(args[1], int)
            and cast(int, args[1]) > 1
            and cast(int, args[1]) < 5
            and op
            in {
                exir_ops.edge.aten.pow.Tensor_Scalar,
            }
        ):
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        exponent = cast(int, args[1])

        if exponent > 2:
            for _ in range(exponent, 2, -1):
                x = super().call_operator(
                    exir_ops.edge.aten.mul.Tensor,
                    (x, args[0]),
                    {},
                    meta,
                )
        return super().call_operator(
            exir_ops.edge.aten.mul.Tensor,
            (x, args[0]),
            {},
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceMatmulWithTransposedMatmulPass(ExportPass):
    """
    For certain backends, we have efficient kernels for transposed matmul. We
    replace AxB with AxB' for such backends.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.cadence.quantized_matmul.default or args[-1] is True:
            return super().call_operator(op, args, kwargs, meta)

        # Get the args
        if len(args) == 9:
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
            ) = args
        elif len(args) == 8:
            (
                X_arg,
                X_zero_point,
                Y_arg,
                Y_zero_point,
                bias,
                out_multiplier,
                out_shift,
                out_zero_point,
            ) = args
            transposed = False
        else:
            raise AssertionError(
                f"Unexpected number of args for quantized_matmul: {len(args)}"
            )

        # If the matmul is already transposed, bail
        if transposed:
            return super().call_operator(op, args, kwargs, meta)

        # Get the second tensor
        Y_tensor = Y_arg.to_tensor()
        # Concretize the bias
        zero_bias = super().call_operator(
            exir_ops.edge.aten.full.default,
            ([Y_tensor.size(-1)], 0),
            {"dtype": torch.int32},
            meta,
        )

        # Y_arg is always a ProxyValue, so we insert a transpose node
        transpose_args = (Y_arg, -1, -2)
        Y_arg_t = super().call_operator(
            exir_ops.edge.aten.transpose_copy.int,
            transpose_args,
            {},
            meta,
        )

        # Construct the new args, and return the transposed matmult op
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
        return super().call_operator(op, new_args, kwargs, meta)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        result = super().call(graph_module)
        # Fuse any inserted transpose node with transpose/permute nodes
        # surrounding it.
        result = FuseCascadedTransposeOrPermuteOps()(result.graph_module)
        assert result is not None
        # Replace permute with transpose.
        result = ReplacePermuteWithTransposePass()(result.graph_module)
        assert result is not None
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceMulTensorWithMulAndFullOpsPass(ExportPass):
    """
    Extracts a single value argument of mul op to a separate full op.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for mul_node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mul.Tensor
        ):
            x_arg, const_arg = mul_node.args

            # Swap arguments if the order is wrong
            if isinstance(const_arg, torch.fx.Node):
                x_arg, const_arg = const_arg, x_arg

            # Skip if the const_arg is not a scalar
            if not isinstance(const_arg, (float, int)) or not isinstance(
                x_arg, torch.fx.Node
            ):
                continue

            # Cast the const_arg to the dtype of the x_arg
            full_arg = self.resolve_full_arg(x_arg, const_arg)

            full_output_dtype = (
                torch.int32 if isinstance(full_arg, int) else torch.float32
            )

            # Extract an argument to a separate full op.
            with graph_module.graph.inserting_before(mul_node):
                full_node = graph_module.graph.call_function(
                    torch.ops.aten.full.default,
                    args=([1], full_arg),
                    kwargs={"dtype": full_output_dtype},
                )
                full_node.meta = mul_node.meta
                full_node.meta["val"] = [1]
                new_mul_node = graph_module.graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(x_arg, full_node)
                )
                new_mul_node.meta = mul_node.meta
            # Replace the old mul with a newly created mul.
            mul_node.replace_all_uses_with(new_mul_node)
            graph_module.graph.erase_node(mul_node)
        return super().call(graph_module)

    def resolve_full_arg(self, x_arg, const_arg):
        if x_arg.meta["val"].dtype == torch.float32 and isinstance(const_arg, int):
            const_arg = float(const_arg)
        if x_arg.meta["val"].dtype == torch.int32 and isinstance(const_arg, float):
            const_arg = int(const_arg)
        return const_arg


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass(ExportPass):
    """
    Replace the aten adaptive avg_pool op with the aten avg_pool2d op.
    """

    def call_operator(self, op, args, kwargs, meta):
        # Only continue for avg_pool op
        if op not in {exir_ops.edge.aten._adaptive_avg_pool2d.default}:
            return super().call_operator(op, args, kwargs, meta)

        # Get the input tensor
        in_tensor = args[0].to_tensor()
        # Permute NCHW to NHWC for computation
        in_tensor_permuted = in_tensor.permute(0, 2, 3, 1)
        in_tensor_shape = in_tensor_permuted.shape

        output_size = args[1]
        num_dims = len(output_size)

        # TODO: If in_tensor_shape is not a multiple of output size,
        # this pass will not work. T224984800
        dim_multiples = [
            (in_tensor_shape[i + 1] % output_size[i]) == 0 for i in range(num_dims)
        ]
        if not all(dim_multiples):
            logging.info(
                f"Unable to replace adaptive average pool with average pool. Input tensor shape of {in_tensor_shape} is not a multiple of output size: {output_size}"
            )
            return super().call_operator(op, args, kwargs, meta)

        # Compute stride and kernel_size, then set default values for other arguments
        stride = [(in_tensor_shape[i + 1] // output_size[i]) for i in range(num_dims)]
        kernel_size = [
            in_tensor_shape[i + 1] - (output_size[i] - 1) * stride[i]
            for i in range(num_dims)
        ]
        padding = [0] * num_dims
        ceil_mode = False
        count_include_pad = True
        divisor_override = None

        # Create a new avg_pool node with the updated args
        new_args = (
            args[0],
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
        return super().call_operator(
            exir_ops.edge.aten.avg_pool2d.default,
            new_args,
            kwargs,
            meta,
        )


class CommonReplacePasses:
    passes = [
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
    ]


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAtenLinalgSvdWithCadenceLinalgSvdPass(ExportPass):
    """
    Replace aten linalg svd op with cadence custom op.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten._linalg_svd.default:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.linalg_svd.default, args, kwargs, meta
        )


# This class encapsulates all the functions that replace/switch one op in the
# graph with another.
class CadenceReplaceOpsInGraph:
    passes = CommonReplacePasses.passes + [
        ReplaceAtenLinalgSvdWithCadenceLinalgSvdPass,
        ReplaceEmptyTensorsWithFullPass,
        ReplaceFunctionallyEquivalentOpTargets,
        ReplacePermuteWithTransposePass,
        ReplaceScalarWithTensorArgPass,
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
        ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass,
        ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass,
        ReplaceAtenAvgPoolWithCadenceAvgPoolPass,
        ReplaceWhereWithFullArgsWithWhereScalar,
        ReplaceAtenApproxGeluWithApproxGeluPass,
        ReplaceMulTensorWithMulAndFullOpsPass,
    ]
