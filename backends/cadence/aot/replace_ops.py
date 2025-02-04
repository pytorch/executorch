# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


# This file contains all the functions that replace one op with another in the
# graph. The functions replacing ops for models deployed with Jarvis are grouped
# together in class 'ReplaceOpsInGraph'. Some examples of functions in the class are
# 1. functions that replace an ATen op with a custom op that accepts extra arguments
# 2. functions that replace in-place variants of ATen ops with out-of-place version.
# 3. functions that replace an ATen op with another semantically equivalent ATen op.
# 4. functions that concretize optional args.

# pyre-unsafe

import math
from operator import neg
from typing import cast, Dict, Iterable, Sequence, Set, Tuple

import torch
import torch.fx
from executorch.backends.cadence.aot.compiler_utils import (
    get_shape,
    get_tensor_from_attr,
    get_transposed_dims,
    get_zero_point,
    is_node_with_op,
    is_quantized_tensor,
    quantize_tensor_multiplier,
)
from executorch.backends.cadence.aot.fuse_ops import FuseCascadedViewOps
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
)
from executorch.backends.cadence.aot.remove_ops import RemoveNopSelectOpPass
from executorch.backends.cadence.aot.utils import get_edge_overload_packet
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import ExportPass, NodeMetadata, PassResult, ProxyValue
from torch._subclasses import FakeTensor
from torch.fx.node import Argument

# A map to represent ops that:
# (a) are functionally equivalent wrt. Jarvis; and
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
            logical_not_input_tensor = (
                logical_not_node.args[0].to_tensor()
                if isinstance(logical_not_node.args[0], ProxyValue)
                else logical_not_node.args[0]
            )

            # If the logical_not input is not a boolean tensor, bail.
            if logical_not_input_tensor.meta["spec"].dtype != torch.bool:
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
        if op not in {exir_ops.edge.quantized_decomposed.quantize_per_tensor.default}:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.quantize_per_tensor.default,
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
        if op not in {exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default}:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.cadence.dequantize_per_tensor.default,
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
        in_tensor = args[0].to_tensor() if isinstance(args[0], ProxyValue) else args[0]
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
class ReplaceTCopyWithTransposePass(ExportPass):
    """
    Replace t_copy with transpose_copy.int. If the input is 1D, the t_copy is
    a nop. t_copy is not supported, so this is an opt_level=0 pass.
    """

    def call_operator(self, op, args, kwargs, meta):
        if get_edge_overload_packet(op) != exir_ops.edge.aten.t_copy:
            return super().call_operator(op, args, kwargs, meta)

        # Get the input tensor shape
        in_tensor = args[0].to_tensor() if isinstance(args[0], ProxyValue) else args[0]

        # If the input is a 1D tensor, this t_copy is a nop, so return the input
        if in_tensor.dim() <= 1:
            return args[0]

        assert in_tensor.dim() == 2, "t_copy expects a tensor with <= 2 dimensions"
        transpose_args = (args[0], 0, 1)
        return super().call_operator(
            exir_ops.edge.aten.transpose_copy.int, transpose_args, kwargs, meta
        )


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
        mat2_tensor = mat2.to_tensor() if isinstance(mat2, ProxyValue) else mat2
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
        in_tensor = args[0].to_tensor() if isinstance(args[0], ProxyValue) else args[0]
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
        if get_edge_overload_packet(op) != exir_ops.edge.aten.convolution:
            return super().call_operator(op, args, kwargs, meta)

        # Check if the bias is already concrete
        assert len(args) == 9
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
        in_shape = list(
            in_tensor.to_tensor().shape
            if isinstance(in_tensor, ProxyValue)
            else in_tensor.shape
        )

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
class ReplaceAtenConvolutionWithJarvisConvolutionPass(ExportPass):
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
            transposed_weight = (
                super().call_operator(
                    exir_ops.edge.aten.transpose_copy.int,
                    (
                        weight,
                        0,
                        1,
                    ),
                    kwargs,
                    meta,
                )
                if isinstance(weight, ProxyValue)
                else weight.transpose(0, 1)
            )

            flipped_weight = (
                super().call_operator(
                    exir_ops.edge.aten.flip.default,
                    (
                        transposed_weight,
                        [-1] if transposed_weight.to_tensor().dim() == 3 else [-1, -2],
                    ),
                    kwargs,
                    meta,
                )
                if isinstance(transposed_weight, ProxyValue)
                else (
                    transposed_weight.flip(-1)
                    if transposed_weight.dim() == 3
                    else transposed_weight.flip(-1, -2)
                )
            )

            # From the previous checks, if flipped_weight is a FakeTensor, it has to be
            # a constant (if not, it would be a ProxyValue). Mark it as such.
            if isinstance(flipped_weight, FakeTensor):
                flipped_weight.constant = flipped_weight
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
            ), "Cannot handle padded output in convolution"

            # If the innermost dim of output tensor is 1, then the stride
            # should be 1. Note that the first dimension of output tensor is
            # channel
            new_stride = stride.copy()
            out_shape = meta["val"].shape
            assert out_shape is not None
            for i, e in enumerate(out_shape[2:]):
                new_stride[i] = 1 if e == 1 else stride[i]

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


# TODO(matthiascremon): this is a fuse op, not a replace op
class ReplaceConvWithChannelLastConv:
    """
    Convolution op in pytorch expects NCHW layout for input, weight, and output
    tensors. However, if the input and output to the convolution op are originally
    in NWHC layout, and are then permuted to conform to NCHW layout, we can fuse
    the two permute ops with the convolution op, and call the NHWC layout
    convolution op in Jarvis.
    """

    def __init__(self):
        self.counter = 0
        self.graph_module = None

    def __call__(self, graph_module: torch.fx.GraphModule):
        self.replace_conv_with_nhwc_conv(graph_module)

    def conv_layout_is_nhwc(self, node: torch.fx.Node) -> bool:
        """
        Return true if the convolution input and output are connected to permute
        ops, and the input/output to/from the permute ops is NHWC layout tensor.
        """
        # There must only be a single user of the output node (which must be a
        # permute/tranpsose op). The input of the convolution must be connected
        # to a permute op, and that permute op should have a single user.
        conv_inp = node.args[0]
        assert isinstance(conv_inp, torch.fx.Node)
        if len(node.users) != 1 or len(conv_inp.users) != 1:
            return False

        # Get the input and output (permute/transpose) nodes of the convolution
        conv_user = list(node.users.keys())[0]
        assert isinstance(conv_user, torch.fx.Node)
        pt_nodes: Set[torch.fx.Node] = {conv_inp, conv_user}

        # Any node in pt_nodes must not be a placeholder.
        if contains_placeholder_or_param(pt_nodes):
            return False

        # Determine if the convolution is 1d or 2d. The output tensor must be
        # 3- or 4-dimensional
        out_shape = get_shape(self.graph_module, node)
        assert out_shape is not None
        out_dims = len(out_shape)
        assert out_dims in {3, 4}, "Jarvis only supports conv1d and conv2d"
        conv1d = out_dims == 3

        # Get the possible targets for the nodes in pt_nodes. Since conv1d has
        # 3-dimensional input and output tensors, the nodes in pt_nodes could
        # be either permute or transpose op. For conv2d, the nodes in pt_nodes
        # must be permute ops.
        p_target = exir_ops.edge.aten.permute_copy.default
        t_target = exir_ops.edge.aten.transpose_copy.int
        pt_targets = [p_target] + ([t_target] if conv1d else [])

        # If any node in pt_nodes is not permute op (or tranpose op for conv1d),
        # bail.
        if any(x.target not in pt_targets for x in pt_nodes):
            return False

        # Now we need to determine the dimension permutations:
        # If the input had NHWC layout, which was then permuted/transposed
        # by a permute/transpose op to NCHW layout, the permutation must be
        # [0, 3, 2, 1] (or [0, 2, 1] for conv1d).
        # If the output had NCHW layout, and was then permuted to NHWC layout,
        # the permutation must be [0, 2, 3, 1] (or [0, 2, 1] for conv1d).
        nhwc_permute_order = {
            node.args[0]: [0, 2, 1] if conv1d else [0, 3, 1, 2],
            list(node.users.keys())[0]: [0, 2, 1] if conv1d else [0, 2, 3, 1],
        }
        for x in pt_nodes:
            order = (
                x.args[1]
                if x.target == p_target
                else get_transposed_dims(x, list(range(out_dims)))
            )
            if order != nhwc_permute_order[x]:
                return False

        return True

    def replace_conv_with_nhwc_conv(self, graph_module: torch.fx.GraphModule):
        self.graph_module = graph_module
        graph = graph_module.graph
        for node in graph.nodes:
            # We are only interested in convolution nodes that have NHWC layout
            if node.target not in {
                exir_ops.edge.cadence.quantized_conv.default,
                exir_ops.edge.cadence.convolution.default,
                exir_ops.edge.cadence.quantized_transposed_conv.default,
                exir_ops.edge.cadence.transposed_convolution.default,
            } or not self.conv_layout_is_nhwc(node):
                continue

            # Get the args of convolution op
            args = list(node.args)
            # The input is connected to a permute/transpose op that converts the
            # NHWC layout to NCHW layout. The input of the permute op will become
            # this convolution op's input.
            in_tp = args[0]
            args[0] = in_tp.args[0]
            # The weight is in NHWC layout. Permute it to NHWC layout.
            weight_tensor = get_tensor_from_attr(graph_module, args[1])
            assert isinstance(weight_tensor, torch.Tensor)
            # We cannot directly permute a per-channel quantized tensor. We will
            # dequantize it, permute the fp32 tensor, and then requantize the
            # permuted tensor.
            if (
                is_quantized_tensor(weight_tensor)
                and weight_tensor.qscheme() == torch.per_channel_affine
            ):
                # We have already asserted during quantizing conv op that the
                # quantization axis is 0.
                dequant_weight = weight_tensor.dequantize()
                dequant_weight = (
                    dequant_weight.permute([0, 2, 1])
                    if dequant_weight.dim() == 3
                    else dequant_weight.permute([0, 2, 3, 1])
                )
                weight_tensor = torch.quantize_per_channel(
                    dequant_weight.contiguous(),
                    weight_tensor.q_per_channel_scales(),
                    weight_tensor.q_per_channel_zero_points(),
                    0,
                    weight_tensor.dtype,
                )
            else:
                weight_tensor = (
                    weight_tensor.permute([0, 2, 1])
                    if weight_tensor.dim() == 3
                    else weight_tensor.permute([0, 2, 3, 1])
                )
            # Make the weight tensor contiguous, since we have permuted it.
            weight_tensor = weight_tensor.contiguous()
            # Add the permuted weight into the graph, and update the weight in
            # args.
            with graph.inserting_before(node):
                weight_name = f"_weight_nhwc_{self.counter}"
                graph_module.register_buffer(weight_name, weight_tensor)
                weight = graph.get_attr(weight_name)
            args[1] = weight

            # The 'channel_last' arg is True. It is the last arg.
            args[-1] = True
            # Now update the convolution node args to mark it as NHWC convolution
            node.args = tuple(args)

            # Replace all the uses of the permute op connected to the output op
            # with this convolution.
            out_tp = list(node.users.keys())[0]
            out_tp.replace_all_uses_with(node)
            node.meta = out_tp.meta

            # Erase the permute ops connected to the input and output of the
            # convolution op.
            graph.erase_node(in_tp)
            graph.erase_node(out_tp)
            self.counter += 1

        graph_module.recompile()


# This pass needs to be reworked to be compatible with PT2. It is an optimization
# pass anyway, so move it to opt level 2.
# TODO(matthiascremon): update and improve this pass.
@register_cadence_pass(CadencePassAttribute(opt_level=2))
class ReplaceConvWithChannelLastConvPass(ExportPass):
    """
    Replace the ATen convolution op with custom conv op with NCHW or NHWC layout
    input tensors, depending on the presence of permute/transpose ops connected
    to the input tensor.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        result = ReplaceAtenConvolutionWithJarvisConvolutionPass()(graph_module)
        assert result is not None
        ReplaceConvWithChannelLastConv()(result.graph_module)
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
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
        exir_ops.edge.cadence.quantized_conv.default: exir_ops.edge.cadence.quantized_linear.default,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.trivial_conv_op_to_linear_op:
            return super().call_operator(op, args, kwargs, meta)

        # Parse the necessary args of the convolution node. Both convolution
        # and quantized_conv have the same first 8 args. The quantized op has
        # extra args holding at least the zero point and scale of input, weight, bias,
        # and output tensor.
        quantized_op = op == exir_ops.edge.cadence.quantized_conv.default
        assert (len(args) == 8 and not quantized_op) or (
            len(args) >= 12 and quantized_op
        ), "Inconsistent args for convolution"
        (in_tensor, weight, bias, stride, padding, dilation, groups) = args[0:7]

        # Glean the shapes of input, weight, and output
        in_shape = (
            in_tensor.to_tensor().shape
            if isinstance(in_tensor, ProxyValue)
            else in_tensor.shape
        )

        weight_shape = (
            weight.to_tensor().shape if isinstance(weight, ProxyValue) else weight.shape
        )
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

        # If weight is a ProxyValue, linear_weight needs to be the output of a
        # graph operation (in this case a view_copy op) to be an explicit ProxyValue
        # as well. If not, the view op can be done directly on the tensor.
        linear_weight = (
            super().call_operator(
                exir_ops.edge.aten.view_copy.default,
                (
                    weight,
                    [weight_shape[0], K],
                ),
                kwargs,
                meta,
            )
            if isinstance(weight, ProxyValue)
            else weight.contiguous().view(weight_shape[0], K)
        )
        # From the previous check, if linear_weight is a FakeTensor, it has to be
        # a constant (if not, it would be a ProxyValue). Mark it as such.
        if isinstance(linear_weight, FakeTensor):
            linear_weight.constant = linear_weight

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
            if (
                len(args) >= 14
                and isinstance(args[12], ProxyValue)
                and isinstance(args[13], ProxyValue)
            ):
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
class ForceChannelLastForConvPass(ExportPassWithTransposeHelper):
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
            exir_ops.edge.cadence.quantized_conv.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        quantized_op = op == exir_ops.edge.cadence.quantized_conv.default
        channel_last_arg_index = 14 if quantized_op else 7
        channel_last = (
            args[channel_last_arg_index]
            if len(args) > channel_last_arg_index
            # Default is false (NCHW).
            else False
        )
        if channel_last:
            return super().call_operator(op, args, kwargs, meta)

        input_proxy = cast(ProxyValue, args[0])
        weight_proxy = cast(ProxyValue, args[1])
        input_proxy = self.change_nchw_to_nhwc(input_proxy, meta)
        weight_proxy = self.change_nchw_to_nhwc(weight_proxy, meta)

        new_args = (
            # Transposed input/weights.
            (input_proxy, weight_proxy)
            # All other args (bias, quant params, etc)
            + tuple(args[2:channel_last_arg_index])
            # Channel last.
            + (True,)
        )
        output_proxy = super().call_operator(op, new_args, kwargs, meta)
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


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceConvWithIm2RowAndLinear(ExportPass):
    """
    Replace convolution where groups=1 with im2row followed by a linear op.
    """

    # A map from the convolution op to the linear op that it should
    # decompose to.
    conv_op_to_linear_op: Dict[EdgeOpOverload, EdgeOpOverload] = {
        exir_ops.edge.cadence.convolution.default: exir_ops.edge.aten.linear.default,
        exir_ops.edge.cadence.quantized_conv.default: exir_ops.edge.cadence.quantized_linear.default,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.conv_op_to_linear_op:
            return super().call_operator(op, args, kwargs, meta)

        # Get the relevant args from convolution node.
        quantized_op = op == exir_ops.edge.cadence.quantized_conv.default
        assert (len(args) == 8 and not quantized_op) or (
            len(args) >= 12 and quantized_op
        ), "Inconsistent args for convolution"
        (in_tensor, weight, bias, stride, padding, dilation, groups) = args[0:7]

        # We do not replace depthwise convolution with gemm yet.
        if groups != 1:
            return super().call_operator(op, args, kwargs, meta)

        weight_shape = (
            weight.to_tensor().shape if isinstance(weight, ProxyValue) else weight.shape
        )
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
        channel_last = (
            (args[14] if len(args) == 15 else False) if quantized_op else args[-1]
        )
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
                if isinstance(in_tensor.to_tensor(), FakeTensor)
                else get_zero_point(in_tensor.to_tensor())
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

        # If weight is a ProxyValue, linear_weight needs to be the output of a
        # graph operation (in this case a view_copy op) to be an explicit ProxyValue
        # as well. If not, the view op can be done directly on the tensor.
        linear_weight = (
            super().call_operator(
                exir_ops.edge.aten.view_copy.default,
                (
                    weight,
                    [weight_shape[0], K],
                ),
                kwargs,
                meta,
            )
            if isinstance(weight, ProxyValue)
            else weight.contiguous().view(weight_shape[0], K)
        )
        # From the previous check, if linear_weight is a FakeTensor, it has to be
        # a constant (if not, it would be a ProxyValue). Mark it as such.
        if isinstance(linear_weight, FakeTensor):
            linear_weight.constant = linear_weight

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
            if (
                len(args) >= 14
                and isinstance(args[12], ProxyValue)
                and isinstance(args[13], ProxyValue)
            ):
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


@register_cadence_pass(CadencePassAttribute(opt_level=1))
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
        weight_shape = (
            weight.to_tensor().shape if isinstance(weight, ProxyValue) else weight.shape
        )
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

        # If weight is a ProxyValue, linear_weight needs to be the output of a
        # graph operation (in this case a view_copy op) to be an explicit ProxyValue
        # as well. If not, the view op can be done directly on the tensor.
        linear_weight = (
            super().call_operator(
                exir_ops.edge.aten.view_copy.default,
                (
                    weight,
                    [weight_shape[0], K],
                ),
                kwargs,
                meta,
            )
            if isinstance(weight, ProxyValue)
            else weight.contiguous().view(weight_shape[0], K)
        )
        # From the previous check, if linear_weight is a FakeTensor, it has to be
        # a constant (if not, it would be a ProxyValue). Mark it as such.
        if isinstance(linear_weight, FakeTensor):
            linear_weight.constant = linear_weight

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
        in_tensor = args[0].to_tensor() if isinstance(args[0], ProxyValue) else args[0]
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
        result = FuseCascadedViewOps()(result.graph_module)
        assert result is not None
        return result


@register_cadence_pass(CadencePassAttribute(opt_level=1))
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
        in_tensor = args[0].to_tensor() if isinstance(args[0], ProxyValue) else args[0]
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


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceScalarWithTensorArgPass(ExportPass):
    """
    For binary ops like add.Scalar, sub.Scalar mul.Scalar, and div.Scalar,
    replace the scalar arg with Tensor arg.
    """

    scalar_to_tensor_ops: Dict[EdgeOpOverload, EdgeOpOverload] = {
        exir_ops.edge.aten.add.Scalar: exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Scalar: exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Scalar: exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Scalar: exir_ops.edge.aten.div.Tensor,
    }

    def get_replacement(self, op, args, kwargs, meta):
        return super().call_operator(
            # Replace with .Tensor variant.
            op=self.scalar_to_tensor_ops[op],
            args=(
                # Tensor arg.
                args[0],
                # Scalar arg - replace with aten.full tensor.
                super().call_operator(
                    exir_ops.edge.aten.full.default,
                    args=(
                        (1,),
                        args[1],
                    ),
                    kwargs={"dtype": args[0].to_tensor().dtype},
                    meta=meta,
                ),
                # Other args.
                *args[2:],
            ),
            kwargs=kwargs,
            meta=meta,
        )

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.scalar_to_tensor_ops:
            return super().call_operator(op, args, kwargs, meta)

        # There must be exactly 2 args (3 for add and sub containing alpha)
        assert len(args) == 2 or len(args) == 3

        # If there are two args, just replace the op.
        if len(args) == 2:
            return self.get_replacement(op, args, kwargs, meta)

        # In case the op has three args, it must be scalar add/sub op.
        if (
            op not in {exir_ops.edge.aten.add.Scalar, exir_ops.edge.aten.sub.Scalar}
            or "alpha" in kwargs
        ):
            return super().call_operator(op, args, kwargs, meta)

        return self.get_replacement(op, args, kwargs, meta)


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
                (
                    args[0].to_tensor().shape
                    if isinstance(args[0], ProxyValue)
                    else args[0].shape
                ),
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


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class ReplaceAtenLinalgVectorNormWithCadenceLinalgVectorNormPass(ExportPass):
    """
    Replace the aten.linalg_vector_norm op with a custom op.
    aten.linalg_vector_norm is not supported by Jarvis, so we
    need to replace it with native_batch_norm at all optimization levels.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.linalg_vector_norm.default:
            return super().call_operator(op, args, kwargs, meta)

        assert (
            len(args) == 1
        ), "aten.linalg_vector_norm should have 1 argument (a tensor), we do not support any custom variants"

        return super().call_operator(
            exir_ops.edge.cadence.linalg_vector_norm.default,
            args,
            kwargs,
            meta,
        )


@register_cadence_pass(CadencePassAttribute(opt_level=1))
class ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass(ExportPass):
    """
    Replace ops with single element arguments (size = [1]) with overloads that accept scalar ints/floats.
    """

    # Keep track of which operators and arguments are being replaced.
    replaced_scalar_args: dict[
        EdgeOpOverloadPacket, tuple[EdgeOpOverload, Sequence[int]]
    ] = {
        exir_ops.edge.cadence.quantized_conv: (
            exir_ops.edge.cadence.quantized_conv.per_tensor,
            [8, 9, 12, 13],
        ),
        exir_ops.edge.cadence.quantized_fully_connected: (
            exir_ops.edge.cadence.quantized_fully_connected.per_tensor,
            [4, 5, 6],
        ),
        exir_ops.edge.cadence.quantized_layer_norm: (
            exir_ops.edge.cadence.quantized_layer_norm.per_tensor,
            [1, 2],
        ),
        exir_ops.edge.cadence.quantized_linear: (
            exir_ops.edge.cadence.quantized_linear.per_tensor,
            [4, 5, 6],
        ),
        exir_ops.edge.cadence.quantized_relu: (
            exir_ops.edge.cadence.quantized_relu.per_tensor,
            [1, 3, 4],
        ),
    }

    def call_operator(self, op, args, kwargs, meta):
        op_edge_overload_packet = get_edge_overload_packet(op)

        if op_edge_overload_packet not in self.replaced_scalar_args:
            return super().call_operator(op, args, kwargs, meta)

        # Get all the args that need to be replaced.
        new_op, args_to_be_replaced = self.replaced_scalar_args[op_edge_overload_packet]

        updated_args = list(args)
        for op_arg_index in args_to_be_replaced:
            arg = args[op_arg_index]
            if not isinstance(arg, ProxyValue):
                return super().call_operator(op, args, kwargs, meta)

            if not arg.is_tensor():
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
class ReplaceAtenAvgPoolWithJarvisAvgPoolPass(ExportPass):
    """
    Replace the aten avg_pool op with the jarvis custom avg_pool2d op.
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
        in_tensor = args[0].to_tensor() if isinstance(args[0], ProxyValue) else args[0]

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
        zero_point = torch.tensor(0, dtype=torch.int32)

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


# This class encapsulates all the functions that replace/switch one op in the
# graph with another.
class CadenceReplaceOpsInGraph:
    passes = [
        ReplaceFunctionallyEquivalentOpTargets,
        ReplaceTCopyWithTransposePass,
        ReplacePermuteWithTransposePass,
        ReplaceScalarWithTensorArgPass,
        ReplaceConvolutionOptionalArgsWithConcreteArgsPass,
        ReplaceMMWithAddMMPass,
        ReplaceSqueezeAndUnsqueezeWithViewPass,
        ReplaceAddMMWithLinearPass,
        RemoveNopSelectOpPass,
        ReplaceSelectWithViewOpPass,
        ReplaceRepeatWithCatPass,
        ReplacePadWithCatPass,
        ReplaceConstantPadNdWithSlicePass,
        ReplaceConvWithChannelLastConvPass,
        ReplaceAtenConvolutionWithJarvisConvolutionPass,
        ForceChannelLastForConvPass,
        ReplaceTrivialConvWithLinear,
        ReplaceConvWithIm2RowAndLinear,
        ReplaceTransposedConvWithLinearPass,
        # This pass should be after passes that replace conv -> im2row + linear.
        ReplaceIm2RowWithViewPass,
        MakeSliceAndCatDimOutermostPass,
        ReplaceNopTransposeOrPermuteWithViewPass,
        ReplaceLinearWithFullyConnectedOpPass,
        ReplaceScalarTensorWithFullPass,
        ReplaceFullLikeWithFullPass,
        ReplaceInfArgInFullWithValuePass,
        ReplaceLogicalNotBooleanWhereWithWherePass,
        ReplacePT2QuantWithCadenceQuantPass,
        ReplacePT2DequantWithCadenceDequantPass,
        ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass,
        ReplaceAtenAvgPoolWithJarvisAvgPoolPass,
        ReplaceAtenLinalgVectorNormWithCadenceLinalgVectorNormPass,
    ]
