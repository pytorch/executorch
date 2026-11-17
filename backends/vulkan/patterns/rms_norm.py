# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops


_CAST_OPS = {
    exir_ops.edge.aten._to_copy.default,
    exir_ops.edge.aten.to.dtype,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
}


def _skip_casts(node: torch.fx.Node) -> torch.fx.Node:
    """Unwrap chains of dtype-cast nodes to find the underlying value."""
    while node.target in _CAST_OPS:
        arg0 = node.args[0] if node.args else None
        if not isinstance(arg0, torch.fx.Node):
            break
        node = arg0
    # pyre-ignore[7]: node is always a Node; Pyre cannot narrow through loops
    return node


def _cast_chain(node: torch.fx.Node) -> List[torch.fx.Node]:
    """`node` followed by the values it was cast from, outermost first."""
    chain = [node]
    while chain[-1].target in _CAST_OPS and isinstance(
        chain[-1].args[0] if chain[-1].args else None, torch.fx.Node
    ):
        chain.append(chain[-1].args[0])
    return chain


def _tensor_val(node) -> Optional[torch.Tensor]:
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    return val if isinstance(val, torch.Tensor) else None


class RmsNormMatch(PatternMatch):
    """
    Detects the decomposed RMSNorm pattern, including variants where dtype
    casts (to_copy) are inserted around the computation.

    The canonical pattern emitted by the Llama RMSNorm implementation is:

      x_orig (any dtype)
        -> to_copy(fp32) -> x_f32
           -> mul(x_f32, x_f32) or pow(x_f32, 2)
           -> mean(dim=-1 or ndim - 1, keepdim=True)
           -> add.Tensor or add.Scalar(eps) -> rsqrt -> rstd_f32
        -> mul(x_f32, rstd_f32) -> norm_f32
        -> to_copy(orig dtype) -> norm_cast
      weight -> to_copy(orig dtype) -> weight_cast
      -> mul(norm_cast, weight_cast)   ← anchor node

    We look through to_copy nodes when comparing tensor identities so that
    the match also handles fp32-only models where no casts are present.

    aten.rms_norm's decomposition (F.rms_norm, nn.RMSNorm) instead casts the
    weight up and the product back down:

      mul(norm_f32, to_copy(weight, fp32)) -> to_copy(orig dtype)

    The anchor node is the final mul (scale by weight), or that trailing cast
    when there is one. The fused op reads the input, the weight and the output
    in one dtype, and prepacks the weight, so the weight must be a constant.
    """

    def __init__(self, final_mul_node: torch.fx.Node) -> None:  # noqa: C901
        self.anchor_node = final_mul_node
        self.match_found = False
        self.all_nodes = [self.anchor_node]

        # final_mul: mul(normalized_cast, weight_cast)
        # Unwrap casts to reach the underlying norm_mul and weight.
        norm_mul_candidate_raw, self.weight_node = self._identify_norm_mul_and_weight(
            final_mul_node
        )
        norm_mul_node = (
            None
            if norm_mul_candidate_raw is None
            else _skip_casts(norm_mul_candidate_raw)
        )
        if norm_mul_node is None:
            return

        # et_vk.rms_norm prepacks the weight, so it has to be a constant. A
        # computed multiplier (Gemma's `1 + w`, or the rsqrt of a chained
        # weightless norm) is not one. Whether a placeholder is a constant or
        # a graph input is checked by the partitioner and the replacement,
        # which have the exported program.
        weight_chain = _cast_chain(self.weight_node)
        self.weight_node = weight_chain[-1]
        if self.weight_node.op != "placeholder":
            return
        self.required_constant_nodes = [self.weight_node]
        self.all_nodes.extend(weight_chain[:-1])
        self.all_nodes.extend(_cast_chain(norm_mul_candidate_raw)[:-1])

        self.all_nodes.append(norm_mul_node)

        # norm_mul: mul(x_f32, rstd_f32)
        rsqrt_node, x_for_norm = self._identify_rsqrt_and_input(norm_mul_node)
        if rsqrt_node is None:
            return

        self.all_nodes.append(rsqrt_node)

        # rsqrt -> add(mean_sq, eps) -> mean(x_sq, dim=-1, keepdim=True)
        # aten.rms_norm decomposes to add.Scalar and dim=[ndim - 1].
        add_node = self._get_single_arg_node(
            rsqrt_node, exir_ops.edge.aten.rsqrt.default
        )
        if add_node is None or add_node.target not in (
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.add.Scalar,
        ):
            return
        # add computes self + alpha * other.
        alpha = add_node.kwargs.get(
            "alpha", add_node.args[2] if len(add_node.args) > 2 else 1
        )
        if alpha != 1:
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

        self.all_nodes.append(mean_node)

        # Verify mean has keepdim=True and reduces the last dim only.
        if len(mean_node.args) < 3 or not mean_node.args[2]:
            return
        mean_in_val = _tensor_val(mean_node.args[0])
        mean_dims = mean_node.args[1]
        if mean_in_val is None or not isinstance(mean_dims, (list, tuple)):
            return
        ndim = mean_in_val.dim()
        if [d % ndim for d in mean_dims] != [ndim - 1]:
            return

        # mean's input should be x_sq = mul(x, x) or pow(x, 2)
        sq_node = mean_node.args[0]
        if not isinstance(sq_node, torch.fx.Node):
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

        if not isinstance(x_for_norm, torch.fx.Node):
            return
        x_chain = _cast_chain(x_for_norm)

        # A product cast back to the original input dtype is the fused op's
        # output, so an fp16 norm stays fp16 in, fp16 out; the shader
        # accumulates in fp32 regardless.
        out_val = _tensor_val(final_mul_node)
        x_val = _tensor_val(x_chain[-1])
        if out_val is None or x_val is None:
            return
        users = list(final_mul_node.users)
        if (
            out_val.dtype != x_val.dtype
            and len(users) == 1
            and users[0].target in _CAST_OPS
            and _tensor_val(users[0]) is not None
            and _tensor_val(users[0]).dtype == x_val.dtype
        ):
            self.anchor_node = users[0]
            self.all_nodes.append(users[0])
            out_val = _tensor_val(users[0])

        # The canonical input node to expose to the fused op is the original
        # tensor before any upcast that already has the output dtype.
        inputs = [n for n in x_chain if _tensor_val(n) is not None]
        inputs = [n for n in inputs if _tensor_val(n).dtype == out_val.dtype]
        if not inputs:
            return
        self.input_node = inputs[-1]
        # Also collect the intermediate cast nodes so they can be cleaned up
        for cast_node in x_chain[: x_chain.index(self.input_node)]:
            if cast_node not in self.all_nodes:
                self.all_nodes.append(cast_node)

        # The shader reads the weight as a dense [hidden] array of the output
        # dtype.
        weight_val = _tensor_val(self.weight_node)
        if (
            weight_val is None
            or weight_val.dtype != out_val.dtype
            or list(weight_val.shape) != [x_val.shape[-1]]
        ):
            return

        self.match_found = True

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
                and isinstance(weight_candidate_raw, torch.fx.Node)
            ):
                return norm_candidate_raw, weight_candidate_raw

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


def _extract_eps_value(eps_node) -> float:
    if isinstance(eps_node, (int, float)):
        return float(eps_node)
    if isinstance(eps_node, torch.fx.Node) and "val" in eps_node.meta:
        val = eps_node.meta["val"]
        if isinstance(val, torch.Tensor):
            return float(val.item())
        if isinstance(val, (int, float)):
            return float(val)
    raise ValueError(f"Cannot extract epsilon value from {eps_node}")


@register_pattern_replacement("rms_norm")
def replace_rms_norm_with_fused_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: RmsNormMatch,
):
    # The detector cannot tell a constant placeholder from a graph input.
    weight = match.weight_node
    if not utils.is_param_node(ep, weight) or utils.is_mutable_buffer_node(weight, ep):
        return False

    eps_val = _extract_eps_value(match.eps_node)

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

    rms_norm_node.meta["val"] = match.anchor_node.meta["val"]
    match.anchor_node.replace_all_uses_with(rms_norm_node)
