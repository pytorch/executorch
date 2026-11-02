# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm._passes.utils import find_patterns
from executorch.backends.qualcomm.builders.node_visitor import (
    dq_ops,
    QNN_TENSOR_TYPE_MAP,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


def _is_node(node):
    return isinstance(node, torch.fx.Node)


def _is_call(node):
    return _is_node(node) and node.op == "call_function"


def _is_placeholder(node):
    return _is_node(node) and node.op == "placeholder"


def _is_get_attr(node):
    return _is_node(node) and node.op == "get_attr"


def _is_add(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
    ]


def _is_dq(node):
    return _is_call(node) and node.target in dq_ops


def _is_mean(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.mean.dim,
        torch.ops.aten.mean.dim,
    ]


def _is_mul(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.mul.Tensor,
        torch.ops.aten.mul.Tensor,
    ]


def _is_pow(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.pow.Tensor_Tensor,
        exir_ops.edge.aten.pow.Tensor_Scalar,
        torch.ops.aten.pow.Tensor_Scalar,
    ]


def _is_rsqrt(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.rsqrt.default,
        torch.ops.aten.rsqrt.default,
    ]


class RecomposeRmsNorm(ExportPass):
    """
    Merge decomposed operators back to one super node.
    """

    def __init__(self, quantization_capture=False):
        super(RecomposeRmsNorm, self).__init__()
        self.rms_norm_target = exir_ops.edge.aten.rms_norm.default
        self.skip_targets = [
            exir_ops.edge.aten.to.dtype,
            exir_ops.edge.aten._to_copy.default,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        ]
        self.quantization_capture = quantization_capture
        if quantization_capture:
            self.rms_norm_target = torch.ops.aten.rms_norm.default
            self.skip_targets = [
                torch.ops.aten.to.dtype,
                torch.ops.aten.to.device,
                torch.ops.aten.type_as.default,
            ]

    def _is_cast(self, node, dtype):
        """
        Whether the node is a cast that changes nothing but the dtype, to a
        dtype at least as wide as `dtype` and at most float32. A narrower cast
        rounds the values, which the rebuilt rms_norm in `dtype` would not do,
        and rms_norm computes in float32 at most.
        """
        if not _is_call(node) or node.target not in self.skip_targets:
            return False
        source, cast = node.args[0].meta["val"], node.meta["val"]
        return (
            torch.promote_types(cast.dtype, dtype) == cast.dtype
            and torch.promote_types(cast.dtype, torch.float32) == torch.float32
            and (cast.layout, cast.device, cast.dim_order())
            == (source.layout, source.device, source.dim_order())
        )

    def _strip_casts(self, node, dtype):
        while self._is_cast(node, dtype):
            node = node.args[0]
        return node

    def _squares_the_scaled_tensor(self, pattern, dtype):
        square, scale_mul = pattern[6], pattern[2]
        base = square.args[0]
        if _is_pow(square):
            exponent = square.args[1]
            if _is_node(exponent):
                # LiftConstantScalarOperands may have turned it into a constant.
                constant = exponent.meta["val"].constant
                exponent = None if constant is None else constant.item()
            squares = exponent == 2
        else:
            squares = self._strip_casts(square.args[1], dtype) is self._strip_casts(
                base, dtype
            )
        scaled = [arg for arg in scale_mul.args if arg is not pattern[3]]
        # T5's norm squares a float32 copy of the tensor it scales.
        return (
            squares
            and len(scaled) == 1
            and self._strip_casts(scaled[0], dtype) is self._strip_casts(base, dtype)
        )

    def _get_norm_input(self, pattern, gamma_node):
        """
        The tensor to rebuild the norm on, or None if it has to stay
        decomposed. rms_norm returns its input's dtype, so the input has to be
        in the pattern's output dtype. Unless the norm is about to be
        quantized, QNN also needs a type for that dtype.
        """
        last_mul_node = pattern[0]
        dtype = last_mul_node.meta["val"].dtype
        if not self._squares_the_scaled_tensor(pattern, dtype):
            return None
        input_node = pattern[6].args[0]
        while self._is_cast(input_node, dtype) and (
            input_node.meta["val"].dtype != dtype
            or input_node.args[0].meta["val"].dtype == dtype
        ):
            input_node = input_node.args[0]
        if input_node.meta["val"].dtype != dtype or (
            not self.quantization_capture and dtype not in QNN_TENSOR_TYPE_MAP
        ):
            return None
        # The "*" in the patterns can match any op, but only casts may sit
        # between the normalization and the weight.
        normalized = next(
            (arg for arg in last_mul_node.args if arg is not gamma_node), None
        )
        return (
            input_node if self._strip_casts(normalized, dtype) is pattern[2] else None
        )

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        # Root Mean Square normalization math equivalent implementation
        patterns = [
            # transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
            [_is_mul, "*", _is_mul, _is_rsqrt, _is_add, _is_mean, _is_pow],
            # executorch.examples.models.llama.norm.RMSNorm
            [_is_mul, "*", _is_mul, _is_rsqrt, _is_add, _is_mean, _is_mul],
        ]

        for node in graph.nodes:
            if not _is_mul(node):
                continue

            rms_norm_patterns = [
                pattern
                for pattern in find_patterns(node, patterns)
                if pattern is not None
            ]

            if len(rms_norm_patterns) > 0:
                # Use first matched pattern
                rms_norm_pattern = rms_norm_patterns[0][0]
                last_mul_node = rms_norm_pattern[0]
                gamma_node = None
                # weight should be a constant
                for arg in last_mul_node.args:
                    if _is_get_attr(arg) or _is_placeholder(arg) or _is_dq(arg):
                        gamma_node = arg

                if gamma_node is None:
                    continue

                input_node = self._get_norm_input(rms_norm_pattern, gamma_node)
                if input_node is None:
                    continue
                # Only the last-dim form the RmsNorm builder supports is recomposed.
                # Anything else, e.g. a 0-d scale, is left decomposed.
                input_shape = list(input_node.meta["val"].shape)
                rank = len(input_shape)
                mean_node = rms_norm_pattern[5]
                mean_dims = mean_node.args[1] or range(rank)
                keepdim = len(mean_node.args) > 2 and mean_node.args[2]
                if (
                    rank == 0
                    or [dim % rank for dim in mean_dims] != [rank - 1]
                    or not (keepdim or rank == 1)
                    or list(gamma_node.meta["val"].shape) != input_shape[-1:]
                ):
                    continue

                eps = rms_norm_pattern[4].args[1]
                if isinstance(eps, torch.fx.Node):
                    eps = eps.meta["val"].constant.item()

                with graph.inserting_before(last_mul_node):
                    # args schema
                    # (Tensor input, int[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
                    rms_node = graph.create_node(
                        "call_function",
                        self.rms_norm_target,
                        (
                            input_node,
                            input_shape[-1:],
                            gamma_node,
                            eps,
                        ),
                    )
                    users = last_mul_node.users.copy()
                    for user in users:
                        user.replace_input_with(last_mul_node, rms_node)
                    # copy metadata
                    rms_node.meta = last_mul_node.meta

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
