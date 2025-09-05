# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm._passes.utils import find_patterns
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


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
        ]
        self.quantization_capture = quantization_capture
        if quantization_capture:
            self.rms_norm_target = torch.ops.aten.rms_norm.default
            self.skip_targets = [
                torch.ops.aten.to.dtype,
            ]

    def _get_input_node(self, node):
        input_node = node.args[0]
        while input_node.target in self.skip_targets:
            input_node = input_node.args[0]
        return input_node

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

                eps = rms_norm_pattern[4].args[1]
                if isinstance(eps, torch.fx.Node):
                    eps = eps.meta["val"].constant.item()
                input_node = self._get_input_node(rms_norm_pattern[6])

                with graph.inserting_before(last_mul_node):
                    # args schema
                    # (Tensor input, int[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
                    rms_node = graph.create_node(
                        "call_function",
                        self.rms_norm_target,
                        (
                            input_node,
                            list(gamma_node.meta["val"].shape),
                            gamma_node,
                            eps,
                        ),
                    )
                    users = last_mul_node.users.copy()
                    for user in users:
                        user.replace_input_with(last_mul_node, rms_node)
                    # copy metadata
                    rms_node.meta = last_mul_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
