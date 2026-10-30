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
from executorch.backends.qualcomm.builders.utils import is_parameter
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch._subclasses.fake_tensor import FakeTensor

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

TO_DTYPE_TARGETS = {
    exir_ops.edge.aten.to.dtype,
    torch.ops.aten.to.dtype,
}

DIM_ORDER_COPY_TARGETS = {
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
}

VIEW_TARGETS = {
    exir_ops.edge.aten.view_copy.default,
    torch.ops.aten.view.default,
    torch.ops.aten.view_as.default,
    torch.ops.aten.view_copy.default,
}


def _is_node(node):
    return isinstance(node, torch.fx.Node)


def _is_call(node):
    return _is_node(node) and node.op == "call_function"


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


def _tensor_dtype(node):
    value = node.meta.get("val") if _is_node(node) else None
    return value.dtype if value is not None else None


def _is_cast_wrapper(node):
    if not _is_call(node) or node.target not in (
        TO_DTYPE_TARGETS | DIM_ORDER_COPY_TARGETS
    ):
        return False
    source = node.args[0]
    source_value = source.meta.get("val") if _is_node(source) else None
    result_value = node.meta.get("val")
    if source_value is None or result_value is None:
        return False
    if source_value.shape != result_value.shape:
        return False
    if node.target in DIM_ORDER_COPY_TARGETS:
        dim_order = node.kwargs.get("dim_order")
        if dim_order is not None and list(dim_order) != list(range(source_value.dim())):
            return False
    return True


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
            node_val = node.meta.get("val")
            if isinstance(node_val, FakeTensor):
                value = node_val.constant
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


class RecomposeRmsNorm(ExportPass):
    """
    Merge decomposed operators back to one super node.
    """

    def __init__(self, quantization_capture=False, edge_program=None):
        super(RecomposeRmsNorm, self).__init__()
        self.rms_norm_target = exir_ops.edge.aten.rms_norm.default
        self.quantization_capture = quantization_capture
        self.edge_program = edge_program
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

    def _get_rms_norm_epsilon(self, add, mean):
        if len(add.args) < 2:
            return None
        if add.args[0] is mean:
            eps_arg = add.args[1]
        elif add.args[1] is mean:
            eps_arg = add.args[0]
        else:
            return None
        return _get_scalar(eps_arg, self.lifted_tensor_constants)

    def _is_fp32_upcast(self, node):
        if not _is_cast_wrapper(node):
            return False
        return (
            _tensor_dtype(node.args[0]) in {torch.float16, torch.bfloat16}
            and _tensor_dtype(node) == torch.float32
        )

    def _get_wildcard_path(self, last_mul, outer_mul):
        def find_path(node, remaining):
            if node is outer_mul:
                return []
            if remaining == 0 or not _is_cast_wrapper(node):
                return None
            path = find_path(node.args[0], remaining - 1)
            return None if path is None else [node, *path]

        for arg in last_mul.args:
            path = find_path(arg, 3)
            if path is not None:
                return path
        return None

    def _is_known_output_cast_path(self, path, square):
        if not path:
            return True
        normalized_input = square.args[0]
        if not self._is_fp32_upcast(normalized_input):
            return False
        input_dtype = _tensor_dtype(normalized_input.args[0])
        if _tensor_dtype(path[0]) != input_dtype:
            return False
        return all(
            _tensor_dtype(node.args[0]) in {torch.float32, input_dtype}
            and _tensor_dtype(node) in {torch.float32, input_dtype}
            for node in path
        )

    def _is_last_dim_reduction(self, mean, input_node):
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
        if dims[0] == -1:
            return True
        input_val = input_node.meta.get("val")
        return input_val is not None and dims[0] == input_val.dim() - 1

    def _validate_rms_norm(self, pattern):
        last_mul, _, outer_mul, inverse_sqrt, add, mean, square = pattern
        if not (
            _is_rsqrt(inverse_sqrt)
            or _is_pow_with_exponent(inverse_sqrt, -0.5, self.lifted_tensor_constants)
        ):
            return None

        if _is_mul(square):
            if len(square.args) != 2 or square.args[0] is not square.args[1]:
                return None
        elif not _is_pow_with_exponent(square, 2, self.lifted_tensor_constants):
            return None

        outer_inputs = [arg for arg in outer_mul.args if arg is not inverse_sqrt]
        if len(outer_inputs) != 1 or outer_inputs[0] is not square.args[0]:
            return None

        wildcard_path = self._get_wildcard_path(last_mul, outer_mul)
        if wildcard_path is None or not self._is_known_output_cast_path(
            wildcard_path, square
        ):
            return None

        if not self._is_last_dim_reduction(mean, square.args[0]):
            return None
        alpha = add.kwargs.get("alpha", add.args[2] if len(add.args) > 2 else 1)
        if alpha != 1:
            return None

        eps = self._get_rms_norm_epsilon(add, mean)
        if eps is None or not math.isfinite(eps) or eps < 0:
            return None
        return eps

    def _is_valid_rms_norm(self, pattern):
        return self._validate_rms_norm(pattern) is not None

    def _find_rms_norm_pattern(self, node, patterns):
        for matched_patterns in find_patterns(node, patterns):
            if matched_patterns is None:
                continue
            for candidate in matched_patterns:
                eps = self._validate_rms_norm(candidate)
                if eps is not None:
                    return candidate, eps
                logger.debug(
                    "Rejected RMS norm candidate %s: invalid normalization semantics",
                    candidate[0].name,
                )
        return None

    def _is_static_weight(self, node):
        source = node.args[0] if _is_dq(node) else node
        return _is_get_attr(source) or (
            self.edge_program is not None and is_parameter(source, self.edge_program)
        )

    def _get_static_weight(self, node):
        if self._is_static_weight(node):
            return node
        if not _is_call(node) or node.target not in VIEW_TARGETS:
            return None

        source = node.args[0]
        source_val = source.meta.get("val") if _is_node(source) else None
        node_val = node.meta.get("val")
        if (
            source_val is None
            or node_val is None
            or source_val.shape != node_val.shape
            or not self._is_static_weight(source)
        ):
            return None
        return source

    def _get_rms_norm_weight(self, last_mul_node, outer_mul, input_node):
        for arg in last_mul_node.args:
            if arg is outer_mul or not _is_node(arg):
                continue
            weight = self._get_static_weight(arg)
            if weight is None:
                continue
            arg_val = weight.meta.get("val")
            input_val = input_node.meta.get("val")
            if (
                arg_val is not None
                and input_val is not None
                and len(arg_val.shape) == 1
                and list(arg_val.shape) == [input_val.shape[-1]]
            ):
                return weight, list(arg_val.shape)
        return None, None

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        # Root Mean Square normalization math equivalent implementation
        patterns = [
            # transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm and
            # transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm
            [_is_mul, "*", _is_mul, _is_rsqrt, _is_add, _is_mean, _is_pow],
            # executorch.examples.models.llama.norm.RMSNorm
            [_is_mul, "*", _is_mul, _is_rsqrt, _is_add, _is_mean, _is_mul],
            # transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm
            [_is_mul, "*", _is_mul, _is_pow, _is_add, _is_mean, _is_pow],
        ]

        for node in graph.nodes:
            if not _is_mul(node):
                continue

            match = self._find_rms_norm_pattern(node, patterns)
            if match is None:
                continue
            rms_norm_pattern, eps = match

            last_mul_node = rms_norm_pattern[0]
            outer_mul = rms_norm_pattern[2]
            normalized_input = rms_norm_pattern[-1].args[0]
            input_node = (
                normalized_input.args[0]
                if self._is_fp32_upcast(normalized_input)
                else normalized_input
            )
            gamma_node, gamma_shape = self._get_rms_norm_weight(
                last_mul_node, outer_mul, input_node
            )
            if gamma_node is None:
                logger.debug(
                    "Rejected RMS norm candidate %s: no non-scalar weight",
                    last_mul_node.name,
                )
                continue

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
