# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch

from executorch.backends.vulkan import utils
from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.utils import get_buffer, get_param, is_buffer, is_param


_CAST_OPS = {
    exir_ops.edge.aten._to_copy.default,
    exir_ops.edge.aten.to.dtype,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
}

_ADD_OPS = {
    exir_ops.edge.aten.add.Scalar,
    exir_ops.edge.aten.add.Tensor,
}

_SUPPORTED_DTYPES = {torch.float16, torch.float32}


def _skip_casts(node: torch.fx.Node) -> torch.fx.Node:
    """Unwrap chains of dtype-cast nodes to find the underlying value."""
    while node.target in _CAST_OPS:
        arg0 = node.args[0] if node.args else None
        if not isinstance(arg0, torch.fx.Node):
            break
        node = arg0
    # pyre-ignore[7]: node is always a Node; Pyre cannot narrow through loops
    return node


class RmsNormMatch(PatternMatch):
    """
    Detects the decomposed RMSNorm pattern, including variants where dtype
    casts (to_copy) are inserted around the computation.

    The canonical FP32 pattern is:

      x_orig (any dtype)
        -> to_copy(fp32) -> x_f32
           -> mul(x_f32, x_f32) -> mean(dim=-1, keepdim=True)
           -> add(eps) -> rsqrt -> rstd_f32
        -> mul(x_f32, rstd_f32) -> norm_f32
      weight -> mul(norm_f32, weight)   ← anchor node

    FP16 variants may cast the normalized value before scaling, or cast the
    input and weight to FP32 and cast the scaled result back to FP16. Casts are
    removed only when the input, weight, FP32 compute, and output dtypes match
    that contract.

    The anchor node is the final mul (scale by weight).
    """

    def __init__(self, final_mul_node: torch.fx.Node) -> None:  # noqa: C901
        self.anchor_node = final_mul_node
        self.match_found = False
        self.all_nodes = [self.anchor_node]

        # final_mul: mul(normalized_cast, weight_cast)
        # Unwrap casts to reach the underlying norm_mul and weight.
        norm_mul_node, self.weight_node = self._identify_norm_mul_and_weight(
            final_mul_node
        )
        if norm_mul_node is None:
            return

        self.all_nodes.append(norm_mul_node)

        # norm_mul: mul(x_f32, rstd_f32)
        rsqrt_node, x_for_norm = self._identify_rsqrt_and_input(norm_mul_node)
        if rsqrt_node is None:
            return

        self.all_nodes.append(rsqrt_node)

        # rsqrt -> add(mean_sq, eps) -> mean(x_sq, dim=-1, keepdim=True)
        add_node = self._get_single_arg_node(
            rsqrt_node, exir_ops.edge.aten.rsqrt.default
        )
        if add_node is None or add_node.target not in _ADD_OPS:
            return

        alpha = (
            add_node.args[2]
            if len(add_node.args) > 2
            else add_node.kwargs.get("alpha", 1)
        )
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha != 1:
            return

        self.all_nodes.append(add_node)

        self.eps_node = None
        mean_node = None
        for arg in add_node.args[:2]:
            if (
                isinstance(arg, torch.fx.Node)
                and arg.target == exir_ops.edge.aten.mean.dim
            ):
                mean_node = arg
            else:
                self.eps_node = arg

        if mean_node is None or self.eps_node is None:
            return

        if add_node.target == exir_ops.edge.aten.add.Scalar:
            if isinstance(self.eps_node, bool) or not isinstance(
                self.eps_node, (int, float)
            ):
                return
            if not math.isfinite(float(self.eps_node)) or self.eps_node < 0:
                return
        elif not self._is_scalar_float_tensor(self.eps_node):
            return

        self.all_nodes.append(mean_node)

        # Verify mean reduces exactly the last dimension and keeps it.
        if len(mean_node.args) < 3:
            return
        mean_dims = mean_node.args[1]
        if not mean_node.args[2]:
            return

        # mean's input should be x_sq = mul(x, x) or pow(x, 2)
        sq_node = mean_node.args[0]
        if not isinstance(sq_node, torch.fx.Node):
            return

        sq_val = sq_node.meta.get("val")
        if not isinstance(sq_val, torch.Tensor) or len(sq_val.shape) == 0:
            return
        if not isinstance(mean_dims, (list, tuple)) or len(mean_dims) != 1:
            return
        if mean_dims[0] not in (-1, len(sq_val.shape) - 1):
            return

        self.all_nodes.append(sq_node)

        # Use the fp32 x (x_for_norm) as the canonical fp32 input.
        # Both mul(x,x) and the norm mul should share the same fp32 source.
        x_f32 = (
            _skip_casts(x_for_norm)
            if isinstance(x_for_norm, torch.fx.Node)
            else x_for_norm
        )

        if sq_node.target == exir_ops.edge.aten.mul.Tensor:
            if sq_node.args[0] != sq_node.args[1]:
                return
            sq_input = sq_node.args[0]
            if not isinstance(sq_input, torch.fx.Node):
                return
            if _skip_casts(sq_input) != x_f32 and sq_input != x_for_norm:
                return
        elif sq_node.target == exir_ops.edge.aten.pow.Tensor_Scalar:
            sq_input = sq_node.args[0]
            if not isinstance(sq_input, torch.fx.Node):
                return
            if _skip_casts(sq_input) != x_f32 and sq_input != x_for_norm:
                return
            if sq_node.args[1] != 2 and sq_node.args[1] != 2.0:
                return
        else:
            return

        # The canonical input node to expose to the fused op is the original
        # tensor before any fp32 upcast (i.e. the input to the first to_copy).
        # If there's no cast, x_for_norm is already the original input.
        self.input_node = (
            _skip_casts(x_for_norm)
            if isinstance(x_for_norm, torch.fx.Node)
            else x_for_norm
        )
        # Also collect the intermediate cast nodes so they can be cleaned up
        cast_node = x_for_norm
        while (
            isinstance(cast_node, torch.fx.Node)
            and cast_node.target in _CAST_OPS
            and cast_node not in self.all_nodes
        ):
            self.all_nodes.append(cast_node)
            cast_node = cast_node.args[0] if cast_node.args else cast_node

        if not self._has_supported_tensor_contract(x_for_norm):
            return

        self.match_found = True

    def _is_scalar_float_tensor(self, node) -> bool:
        if not isinstance(node, torch.fx.Node):
            return False
        val = node.meta.get("val")
        return (
            isinstance(val, torch.Tensor)
            and val.numel() == 1
            and val.dtype in _SUPPORTED_DTYPES
        )

    def _has_supported_tensor_contract(self, x_for_norm) -> bool:
        if not all(
            isinstance(node, torch.fx.Node)
            for node in (self.input_node, x_for_norm, self.weight_node)
        ):
            return False

        input_val = self.input_node.meta.get("val")
        compute_val = x_for_norm.meta.get("val")
        weight_val = self.weight_node.meta.get("val")
        scale_weight_val = self.weight_for_scale_node.meta.get("val")
        norm_for_scale_val = self.norm_for_scale_node.meta.get("val")
        scaled_val = self.anchor_node.meta.get("val")
        if not all(
            isinstance(val, torch.Tensor)
            for val in (
                input_val,
                compute_val,
                weight_val,
                scale_weight_val,
                norm_for_scale_val,
                scaled_val,
            )
        ):
            return False

        if (
            input_val.dtype not in _SUPPORTED_DTYPES
            or compute_val.dtype != torch.float32
            or weight_val.dtype != input_val.dtype
            or scale_weight_val.dtype != scaled_val.dtype
            or norm_for_scale_val.dtype != scaled_val.dtype
            or len(input_val.shape) == 0
            or len(weight_val.shape) != 1
            or weight_val.shape[0] != input_val.shape[-1]
            or input_val.shape != scaled_val.shape
        ):
            return False

        self.output_node = self.anchor_node
        output_val = scaled_val
        if scaled_val.dtype != input_val.dtype:
            if input_val.dtype != torch.float16 or scaled_val.dtype != torch.float32:
                return False
            users = list(self.anchor_node.users)
            if len(users) != 1 or users[0].target not in _CAST_OPS:
                return False
            self.output_node = users[0]
            output_val = self.output_node.meta.get("val")
            if not isinstance(output_val, torch.Tensor):
                return False
            self.all_nodes.append(self.output_node)

        return (
            output_val.dtype == input_val.dtype and input_val.shape == output_val.shape
        )

    def _identify_norm_mul_and_weight(self, final_mul_node):
        """From mul(norm_cast, weight_cast), unwrap casts and find the
        underlying norm-mul node and the weight source node."""
        if len(final_mul_node.args) < 2:
            return None, None

        a, b = final_mul_node.args[0], final_mul_node.args[1]

        for norm_candidate_raw, weight_candidate_raw in [(a, b), (b, a)]:
            if not isinstance(norm_candidate_raw, torch.fx.Node):
                continue
            norm_candidate = _skip_casts(norm_candidate_raw)
            if (
                isinstance(norm_candidate, torch.fx.Node)
                and norm_candidate.target == exir_ops.edge.aten.mul.Tensor
                and self._has_rsqrt_ancestor(norm_candidate)
            ):
                if not isinstance(weight_candidate_raw, torch.fx.Node):
                    return None, None
                self.norm_for_scale_node = norm_candidate_raw
                self.weight_for_scale_node = weight_candidate_raw
                return norm_candidate, _skip_casts(weight_candidate_raw)

        return None, None

    def _has_rsqrt_ancestor(self, mul_node):
        """Check if one of mul_node's args is an rsqrt node (possibly through casts)."""
        for arg in mul_node.args[:2]:
            if not isinstance(arg, torch.fx.Node):
                continue
            if _skip_casts(arg).target == exir_ops.edge.aten.rsqrt.default:
                return True
        return False

    def _identify_rsqrt_and_input(self, norm_mul_node):
        """From mul(x, rstd), find the rsqrt node and the input x.
        The rsqrt may be wrapped in a cast node."""
        if len(norm_mul_node.args) < 2:
            return None, None

        a, b = norm_mul_node.args[0], norm_mul_node.args[1]

        for rsqrt_candidate_raw, input_candidate in [(a, b), (b, a)]:
            if not isinstance(rsqrt_candidate_raw, torch.fx.Node):
                continue
            rsqrt_candidate = _skip_casts(rsqrt_candidate_raw)
            if (
                isinstance(rsqrt_candidate, torch.fx.Node)
                and rsqrt_candidate.target == exir_ops.edge.aten.rsqrt.default
            ):
                return rsqrt_candidate, input_candidate

        return None, None

    def _get_single_arg_node(self, node, expected_target):
        """Get the single input arg of a unary op node."""
        if node.target != expected_target:
            return None
        if len(node.args) < 1 or not isinstance(node.args[0], torch.fx.Node):
            return None
        return node.args[0]


@register_pattern_detector("rms_norm")
def find_rms_norm_patterns(
    node: torch.fx.Node,
) -> Optional[RmsNormMatch]:
    if node.target != exir_ops.edge.aten.mul.Tensor:
        return None

    matched_pattern = RmsNormMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


##
## Pattern Replacement
##


def _validate_eps_value(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"RMSNorm epsilon must be a numeric scalar, got {value}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"RMSNorm epsilon must be finite and nonnegative, got {value}")
    return value


def _extract_eps_value(ep: ExportedProgram, eps_node) -> float:
    if isinstance(eps_node, (int, float)):
        return _validate_eps_value(eps_node)

    tensor = None
    if isinstance(eps_node, torch.fx.Node):
        if is_param(ep, eps_node):
            tensor = get_param(ep, eps_node)
        elif is_buffer(ep, eps_node):
            tensor = get_buffer(ep, eps_node)
        elif utils.is_constant(ep, eps_node):
            constant_name = ep.graph_signature.inputs_to_lifted_tensor_constants[
                eps_node.name
            ]
            tensor = ep.constants.get(constant_name)
        elif utils.is_get_attr_node(eps_node):
            tensor = getattr(ep.graph_module, eps_node.target, None)

    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.numel() != 1
        or tensor.dtype not in _SUPPORTED_DTYPES
    ):
        raise ValueError(f"Cannot extract constant scalar epsilon from {eps_node}")

    return _validate_eps_value(tensor.detach().item())


@register_pattern_replacement("rms_norm")
def replace_rms_norm_with_fused_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: RmsNormMatch,
) -> bool:
    try:
        eps_val = _extract_eps_value(ep, match.eps_node)
    except ValueError:
        return False

    with graph_module.graph.inserting_before(match.anchor_node):
        rms_norm_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.rms_norm.default,
            args=(
                match.input_node,
                match.weight_node,
                eps_val,
            ),
        )

    rms_norm_node.meta["val"] = match.output_node.meta["val"]
    match.output_node.replace_all_uses_with(rms_norm_node)
    return True
