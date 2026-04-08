# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

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
}


def _skip_casts(node: torch.fx.Node) -> torch.fx.Node:
    """Unwrap chains of dtype-cast nodes to find the underlying value."""
    while isinstance(node, torch.fx.Node) and node.target in _CAST_OPS:
        if node.args and isinstance(node.args[0], torch.fx.Node):
            node = node.args[0]
        else:
            break
    return node


class RmsNormMatch(PatternMatch):
    """
    Detects the decomposed RMSNorm pattern, including variants where dtype
    casts (to_copy) are inserted around the computation.

    The canonical pattern emitted by the Llama RMSNorm implementation is:

      x_orig (any dtype)
        -> to_copy(fp32) -> x_f32
           -> mul(x_f32, x_f32) -> mean(dim=-1, keepdim=True)
           -> add(eps) -> rsqrt -> rstd_f32
        -> mul(x_f32, rstd_f32) -> norm_f32
        -> to_copy(orig dtype) -> norm_cast
      weight -> to_copy(orig dtype) -> weight_cast
      -> mul(norm_cast, weight_cast)   ← anchor node

    We look through to_copy nodes when comparing tensor identities so that
    the match also handles fp32-only models where no casts are present.

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
        if add_node is None or add_node.target != exir_ops.edge.aten.add.Tensor:
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

        # Verify mean has keepdim=True and dim=[-1]
        if len(mean_node.args) < 3:
            return
        mean_dims = mean_node.args[1]
        if mean_dims != [-1]:
            return
        if not mean_node.args[2]:
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
            ):
                return norm_candidate, weight_candidate_raw

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
