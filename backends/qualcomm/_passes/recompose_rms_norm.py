# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import numbers

import torch
from executorch.backends.qualcomm._passes.utils import copy_meta, find_patterns
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

logger = logging.getLogger(__name__)

# Export wraps lifted scalar tensor constants before they reach Tensor_Tensor ops.
SCALAR_WRAPPER_TARGETS = {
    exir_ops.edge.aten.detach.default,
    exir_ops.edge.aten.detach_.default,
    exir_ops.edge.aten.lift_fresh_copy.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.detach_.default,
    torch.ops.aten.lift_fresh_copy.default,
}


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
        torch.ops.aten.pow.Tensor_Tensor,
        torch.ops.aten.pow.Tensor_Scalar,
    ]


def _is_rsqrt(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.rsqrt.default,
        torch.ops.aten.rsqrt.default,
    ]


def _get_scalar(value, lifted_tensor_constants=None):
    while _is_node(value) and value.target in SCALAR_WRAPPER_TARGETS:
        value = value.args[0]
    if _is_node(value):
        node = value
        value = (
            lifted_tensor_constants.get(node.name)
            if lifted_tensor_constants is not None
            else None
        )
        if value is None:
            value = getattr(node.meta.get("val"), "constant", None)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return value
    return None


def _is_pow_with_exponent(node, exponent, lifted_tensor_constants=None):
    return _is_pow(node) and (
        _get_scalar(node.args[1], lifted_tensor_constants) == exponent
    )


def _is_pow_neg_half(node, lifted_tensor_constants=None):
    return _is_pow_with_exponent(node, -0.5, lifted_tensor_constants)


class RecomposeRmsNorm(ExportPass):
    """
    Merge decomposed operators back to one super node.
    """

    def __init__(self, quantization_capture=False, edge_program=None):
        super(RecomposeRmsNorm, self).__init__()
        self.rms_norm_target = exir_ops.edge.aten.rms_norm.default
        self.skip_targets = [
            exir_ops.edge.aten.to.dtype,
        ]
        self.quantization_capture = quantization_capture
        self.lifted_tensor_constants = {}
        if edge_program is not None:
            # FP export keeps lifted tensor literals in ExportedProgram.constants,
            # rather than attaching them to the placeholder's FakeTensor metadata.
            for (
                node_name,
                constant_name,
            ) in edge_program.graph_signature.inputs_to_lifted_tensor_constants.items():
                constant = edge_program.constants.get(constant_name)
                if constant is not None:
                    self.lifted_tensor_constants[node_name] = constant
        if quantization_capture:
            self.rms_norm_target = torch.ops.aten.rms_norm.default
            self.skip_targets = [
                torch.ops.aten.to.dtype,
            ]

    def _get_input_node(self, node):
        input_node = node.args[0]
        while input_node.target in self.skip_targets or self._is_fp32_upcast(
            input_node
        ):
            input_node = input_node.args[0]
        return input_node

    @staticmethod
    def _is_fp32_upcast(node):
        # Fold only the no-layout-change fp16-to-fp32 RMSNorm compute upcast.
        # Other dim-order copies may change the input's logical layout.
        if node.target != exir_ops.edge.dim_order_ops._to_dim_order_copy.default:
            return False

        source = node.args[0]
        if not _is_node(source):
            return False
        source_val = source.meta.get("val")
        result_val = node.meta.get("val")
        if source_val is None or result_val is None:
            return False
        if source_val.dtype != torch.float16 or result_val.dtype != torch.float32:
            return False
        if node.kwargs.get("dtype") != torch.float32:
            return False

        dim_order = node.kwargs.get("dim_order")
        return dim_order is None or list(dim_order) == list(range(source_val.dim()))

    def _is_valid_rms_norm(self, pattern):
        _, _, outer_mul, inverse_sqrt, add, mean, square = pattern
        if not (
            _is_rsqrt(inverse_sqrt)
            or _is_pow_neg_half(inverse_sqrt, self.lifted_tensor_constants)
        ):
            return False

        if _is_mul(square):
            if len(square.args) != 2 or square.args[0] is not square.args[1]:
                return False
        elif not _is_pow_with_exponent(square, 2, self.lifted_tensor_constants):
            return False

        outer_inputs = [arg for arg in outer_mul.args if arg is not inverse_sqrt]
        if len(outer_inputs) != 1 or outer_inputs[0] is not square.args[0]:
            return False

        dim = mean.args[1] if len(mean.args) > 1 else mean.kwargs.get("dim")
        keepdim = (
            mean.args[2] if len(mean.args) > 2 else mean.kwargs.get("keepdim", False)
        )
        if isinstance(dim, int):
            dims = [dim]
        elif isinstance(dim, (list, tuple)):
            dims = list(dim)
        else:
            return False
        if len(dims) != 1 or keepdim is not True:
            return False
        if dims[0] != -1:
            input_val = square.args[0].meta.get("val")
            if input_val is None or dims[0] != input_val.dim() - 1:
                return False
        alpha = add.kwargs.get("alpha", add.args[2] if len(add.args) > 2 else 1)
        if alpha != 1:
            return False

        eps = _get_scalar(add.args[1], self.lifted_tensor_constants)
        return eps is not None and math.isfinite(eps) and eps >= 0

    def _find_rms_norm_pattern(self, node, patterns):
        for matched_patterns in find_patterns(node, patterns):
            if matched_patterns is None:
                continue
            candidate = matched_patterns[0]
            if self._is_valid_rms_norm(candidate):
                return candidate
            logger.debug(
                "Rejected RMS norm candidate %s: invalid normalization semantics",
                candidate[0].name,
            )
        return None

    @staticmethod
    def _get_rms_norm_weight(last_mul_node):
        for arg in last_mul_node.args:
            if not (_is_get_attr(arg) or _is_placeholder(arg) or _is_dq(arg)):
                continue
            arg_val = arg.meta.get("val")
            if arg_val is not None and len(arg_val.shape) >= 1:
                return arg, list(arg_val.shape)
        return None, None

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        # Root Mean Square normalization math equivalent implementation
        patterns = [
            # transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
            [_is_mul, "*", _is_mul, _is_rsqrt, _is_add, _is_mean, _is_pow],
            # executorch.examples.models.llama.norm.RMSNorm
            [_is_mul, "*", _is_mul, _is_rsqrt, _is_add, _is_mean, _is_mul],
            # transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm
            [_is_mul, "*", _is_mul, _is_pow, _is_add, _is_mean, _is_pow],
        ]

        for node in graph.nodes:
            if not _is_mul(node):
                continue

            rms_norm_pattern = self._find_rms_norm_pattern(node, patterns)
            if rms_norm_pattern is None:
                continue

            last_mul_node = rms_norm_pattern[0]
            # Export represents model weights as get_attrs or placeholders.
            # Require a non-scalar operand to exclude scalar multipliers.
            gamma_node, gamma_shape = self._get_rms_norm_weight(last_mul_node)
            if gamma_node is None:
                logger.debug(
                    "Rejected RMS norm candidate %s: no non-scalar weight",
                    last_mul_node.name,
                )
                continue

            eps = _get_scalar(rms_norm_pattern[4].args[1], self.lifted_tensor_constants)
            if eps is None:
                logger.debug(
                    "Rejected RMS norm candidate %s: epsilon is not scalar",
                    last_mul_node.name,
                )
                continue
            input_node = self._get_input_node(rms_norm_pattern[6])

            with graph.inserting_before(last_mul_node):
                # args schema
                # (Tensor input, int[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
                rms_node = graph.create_node(
                    "call_function",
                    self.rms_norm_target,
                    (
                        input_node,
                        gamma_shape,
                        gamma_node,
                        eps,
                    ),
                )
                rms_node.meta = copy_meta(last_mul_node.meta)
                users = last_mul_node.users.copy()
                for user in users:
                    user.replace_input_with(last_mul_node, rms_node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
