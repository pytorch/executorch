# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

from executorch.backends.arm._passes import DecomposeSDPAWithRegularSoftmaxPass
from executorch.backends.cortex_m.passes.move_sdpa_scale_after_bmm_pass import (
    MoveSDPAScaleAfterBmmPass,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from torch import nn
from torch.export import export
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


_MUL_SCALAR = torch.ops.aten.mul.Scalar
_BMM = torch.ops.aten.bmm.default
_VIEW = torch.ops.aten.view.default
_EXPAND = torch.ops.aten.expand.default
_SOFTMAX_TARGETS = {
    torch.ops.aten.softmax.int,
    torch.ops.aten._softmax.default,
}


class _SDPA(nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
        )


class _ScaledBmmNoSoftmax(nn.Module):
    """Looks algebraically similar but is not the SDPA score->softmax pattern."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = head_dim**-0.25

    def forward(self, q, k):
        return torch.bmm(
            q * self.scale,
            (k * self.scale).transpose(-2, -1),
        )


def _decomposed_sdpa(
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> GraphModule:
    example_inputs = (
        torch.randn(1, 2, 8, head_dim, dtype=dtype),
        torch.randn(1, 2, 8, head_dim, dtype=dtype),
        torch.randn(1, 2, 8, head_dim, dtype=dtype),
    )

    ep = export(_SDPA().eval(), example_inputs)
    gm = ep.module()

    result = DecomposeSDPAWithRegularSoftmaxPass()(gm)
    assert result is not None
    return result.graph_module


def _count_target(gm: GraphModule, target) -> int:
    return sum(
        node.op == "call_function" and node.target is target for node in gm.graph.nodes
    )


def _find_nodes(gm: GraphModule, target) -> list[Node]:
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is target
    ]


def _trace_back_to_mul(node: object) -> Node | None:
    """Walk backward through the exact expand/view SDPA input path."""

    if not isinstance(node, Node):
        return None

    while node.target in {_VIEW, _EXPAND}:
        if not node.args or not isinstance(node.args[0], Node):
            return None
        node = node.args[0]

    if node.target is _MUL_SCALAR:
        return node

    return None


def _scalar_arg(node: Node, index: int = 1) -> float:
    value = node.args[index]
    assert not isinstance(value, bool)
    assert isinstance(value, (int, float))
    return float(value)


def _find_score_bmm(gm: GraphModule) -> Node:
    """Return the QK^T BMM, identified by its view->softmax consumer."""

    matches = []

    for node in _find_nodes(gm, _BMM):
        users = list(node.users)

        if len(users) != 1:
            continue

        view = users[0]

        if view.target is not _VIEW or len(view.users) != 1:
            continue

        softmax = next(iter(view.users))

        if softmax.target in _SOFTMAX_TARGETS:
            matches.append(node)

    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("head_dim", [16, 128])
def test_moves_default_sdpa_scale_after_score_bmm(head_dim: int):
    torch.manual_seed(head_dim)

    gm = _decomposed_sdpa(head_dim)

    score_bmm_before = _find_score_bmm(gm)

    lhs_mul = _trace_back_to_mul(score_bmm_before.args[0])
    rhs_mul = _trace_back_to_mul(score_bmm_before.args[1])

    assert lhs_mul is not None
    assert rhs_mul is not None

    split_scale = head_dim**-0.25

    assert math.isclose(
        _scalar_arg(lhs_mul),
        split_scale,
        rel_tol=1e-12,
    )
    assert math.isclose(
        _scalar_arg(rhs_mul),
        split_scale,
        rel_tol=1e-12,
    )

    muls_before = _count_target(gm, _MUL_SCALAR)

    result = MoveSDPAScaleAfterBmmPass()(gm)

    assert result is not None
    assert result.modified

    gm = result.graph_module

    # Two split Q/K scales become one score scale.
    assert _count_target(gm, _MUL_SCALAR) == muls_before - 1

    score_bmms = _find_nodes(gm, _BMM)
    score_bmm = None
    score_scale = None

    for bmm in score_bmms:
        users = list(bmm.users)

        if len(users) != 1:
            continue

        candidate = users[0]

        if candidate.target is not _MUL_SCALAR:
            continue

        if len(candidate.users) != 1:
            continue

        view = next(iter(candidate.users))

        if view.target is not _VIEW or len(view.users) != 1:
            continue

        softmax = next(iter(view.users))

        if softmax.target in _SOFTMAX_TARGETS:
            score_bmm = bmm
            score_scale = candidate
            break

    assert score_bmm is not None
    assert score_scale is not None

    combined = _scalar_arg(score_scale)

    assert math.isclose(
        combined,
        1.0 / math.sqrt(head_dim),
        rel_tol=1e-12,
        abs_tol=0.0,
    )

    # Neither BMM operand should trace back to a scalar MUL anymore.
    assert _trace_back_to_mul(score_bmm.args[0]) is None
    assert _trace_back_to_mul(score_bmm.args[1]) is None


def test_does_not_rewrite_non_sdpa_bmm():
    head_dim = 16

    example_inputs = (
        torch.randn(2, 8, head_dim),
        torch.randn(2, 8, head_dim),
    )

    gm = export(
        _ScaledBmmNoSoftmax(head_dim).eval(),
        example_inputs,
    ).module()

    muls_before = _count_target(gm, _MUL_SCALAR)

    result = MoveSDPAScaleAfterBmmPass()(gm)

    assert result is not None
    assert not result.modified
    assert _count_target(result.graph_module, _MUL_SCALAR) == muls_before


def test_does_not_rewrite_non_last_dim_softmax():
    gm = _decomposed_sdpa(16)

    score_bmm = _find_score_bmm(gm)
    score_view = next(iter(score_bmm.users))
    softmax = next(iter(score_view.users))

    assert len(softmax.args) >= 2
    softmax.args = (softmax.args[0], 0, *softmax.args[2:])

    gm.graph.lint()
    gm.recompile()

    result = MoveSDPAScaleAfterBmmPass()(gm)

    assert result is not None
    assert not result.modified


def test_does_not_rewrite_when_scaled_operand_has_multiple_users():
    gm = _decomposed_sdpa(16)

    score_bmm = _find_score_bmm(gm)
    lhs_mul = _trace_back_to_mul(score_bmm.args[0])

    assert lhs_mul is not None
    assert len(lhs_mul.users) == 1

    # Add a second consumer. It does not need to contribute to model output:
    # the matcher sees the extra user before any dead-code elimination.
    with gm.graph.inserting_after(lhs_mul):
        gm.graph.call_function(
            torch.ops.aten.clone.default,
            args=(lhs_mul,),
        )

    gm.graph.lint()
    gm.recompile()

    assert len(lhs_mul.users) == 2

    result = MoveSDPAScaleAfterBmmPass()(gm)

    assert result is not None
    assert not result.modified


def test_does_not_rewrite_wrong_split_scale():
    gm = _decomposed_sdpa(16)

    score_bmm = _find_score_bmm(gm)
    lhs_mul = _trace_back_to_mul(score_bmm.args[0])
    rhs_mul = _trace_back_to_mul(score_bmm.args[1])

    assert lhs_mul is not None
    assert rhs_mul is not None

    lhs_data = lhs_mul.args[0]
    rhs_data = rhs_mul.args[0]

    # Keep the Q/K split scales equal but make them differ from the default
    # SDPA head_dim**(-1/4) factor. This exercises the default-scale guard
    # rather than the unequal-scale guard.
    wrong_scale = _scalar_arg(lhs_mul) * 0.9
    lhs_mul.args = (lhs_data, wrong_scale)
    rhs_mul.args = (rhs_data, wrong_scale)

    gm.graph.lint()
    gm.recompile()

    result = MoveSDPAScaleAfterBmmPass()(gm)

    assert result is not None
    assert not result.modified


def test_does_not_rewrite_when_same_scaled_operand_is_used_twice():
    gm = _decomposed_sdpa(16)

    score_bmm = _find_score_bmm(gm)

    lhs = score_bmm.args[0]
    assert isinstance(lhs, Node)

    # Node.users counts the BMM once even when the same node occupies both
    # argument positions. Without an explicit alias check both matches would
    # refer to the same scalar MUL.
    score_bmm.args = (lhs, lhs)

    gm.graph.lint()
    gm.recompile()

    result = MoveSDPAScaleAfterBmmPass()(gm)

    assert result is not None
    assert not result.modified


def test_rewrites_fp16_sdpa_after_score_path_is_upcast_to_fp32():
    gm = _decomposed_sdpa(
        16,
        dtype=torch.float16,
    )

    score_bmm = _find_score_bmm(gm)

    # PyTorch's math SDPA decomposition promotes the Q/K/V score computation
    # to fp32 before applying the split scale and BMM.
    score_val = score_bmm.meta.get("val")
    assert score_val is not None
    assert score_val.dtype == torch.float32

    muls_before = _count_target(gm, _MUL_SCALAR)
    assert muls_before == 2

    result = MoveSDPAScaleAfterBmmPass()(gm)

    assert result is not None
    assert result.modified

    # Two pre-BMM scales become one post-BMM score scale.
    assert _count_target(result.graph_module, _MUL_SCALAR) == muls_before - 1


def _full_scalar_value(node: Node) -> float | None:
    if (
        node.op != "call_function"
        or node.target is not torch.ops.aten.full.default
        or len(node.args) < 2
    ):
        return None

    shape, value = node.args[:2]

    if not isinstance(shape, (tuple, list, torch.Size)):
        return None

    if tuple(shape) != (1,):
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    return float(value)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0])
    return float(value)


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0])
    return int(value)


@pytest.mark.parametrize("head_dim", [32, 128])
def test_stock_quantizer_produces_fold_compatible_score_qparams(head_dim: int):
    """The registered TFA pass must make the normal quantizer choose
    proportional qparams around the remaining score scale.

    This protects the reason the pass exists, not only its graph rewrite.
    """

    torch.manual_seed(1000 + head_dim)

    example_inputs = (
        torch.randn(1, 2, 8, head_dim),
        torch.randn(1, 2, 8, head_dim),
        torch.randn(1, 2, 8, head_dim),
    )

    exported = export(_SDPA().eval(), example_inputs)
    gm = exported.module()

    result = DecomposeSDPAWithRegularSoftmaxPass()(gm)
    assert result is not None
    gm = result.graph_module

    # Use the ordinary registered CortexMQuantizer.
    prepared = prepare_pt2e(
        gm,
        CortexMQuantizer(),
    )

    # Use several independent calibration batches for the qparam assertion.
    with torch.no_grad():
        for calibration_seed in range(3):
            torch.manual_seed(50000 + head_dim * 100 + calibration_seed)
            prepared(
                torch.randn(1, 2, 8, head_dim),
                torch.randn(1, 2, 8, head_dim),
                torch.randn(1, 2, 8, head_dim),
            )

    converted = convert_pt2e(prepared)

    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dequantize = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    mul_tensor = torch.ops.aten.mul.Tensor

    candidates = []

    for mul in converted.graph.nodes:
        if (
            mul.op != "call_function"
            or mul.target is not mul_tensor
            or len(mul.args) < 2
        ):
            continue

        dq, scalar_node = mul.args[:2]

        if not isinstance(dq, Node) or dq.target is not dequantize:
            continue

        if not isinstance(scalar_node, Node):
            continue

        factor = _full_scalar_value(scalar_node)

        if factor is None:
            continue

        users = list(mul.users)

        if len(users) != 1 or users[0].target is not quantize:
            continue

        q_after = users[0]

        q_before = dq.args[0]

        if not isinstance(q_before, Node) or q_before.target is not quantize:
            continue

        producer = q_before.args[0]

        # Select specifically:
        #
        # BMM -> Q -> DQ -> MUL(score_scale) -> Q
        #
        if not isinstance(producer, Node) or producer.target is not _BMM:
            continue

        s_in = _as_float(dq.args[1])
        zp_in = _as_int(dq.args[2])

        s_out = _as_float(q_after.args[1])
        zp_out = _as_int(q_after.args[2])

        candidates.append(
            (
                factor,
                s_in,
                zp_in,
                s_out,
                zp_out,
            )
        )

    assert len(candidates) == 1

    factor, s_in, zp_in, s_out, zp_out = candidates[0]

    assert math.isclose(
        factor,
        1.0 / math.sqrt(head_dim),
        rel_tol=1e-12,
        abs_tol=0.0,
    )

    assert zp_out == zp_in

    assert math.isclose(
        s_out,
        s_in * factor,
        rel_tol=1e-6,
        abs_tol=0.0,
    )
