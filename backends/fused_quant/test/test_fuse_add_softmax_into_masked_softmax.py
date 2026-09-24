# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import cast

import torch
import torch.fx as fx
from executorch.backends.cadence.aot.compiler import trace
from executorch.backends.fused_quant.pre_quantize_passes.fuse_add_softmax_into_masked_softmax import (
    _has_fully_masked_row,
    _is_additive_mask,
    FuseAddSoftmaxIntoMaskedSoftmax,
)
from parameterized import parameterized
from torch import nn
from torch.export import ExportedProgram

_ADD = torch.ops.aten.add.Tensor
_SAFE_SOFTMAX = torch.ops.aten._safe_softmax.default
_MASKED_SOFTMAX = torch.ops.aten._masked_softmax.default
_INDEX = torch.ops.aten.index.Tensor
_MASK_TYPE_FULL = 2


def _causal_mask(ctx: int) -> torch.Tensor:
    """Binary additive causal mask: 0 on/below the diagonal, -inf above."""
    mask = torch.zeros(ctx, ctx)
    mask.masked_fill_(
        torch.triu(torch.ones(ctx, ctx), diagonal=1).bool(), float("-inf")
    )
    return mask


class _DirectMaskModel(nn.Module):
    """add(scores, mask) -> _safe_softmax, mask used as a direct constant buffer."""

    def __init__(self, ctx: int) -> None:
        super().__init__()
        self.register_buffer("mask", _causal_mask(ctx).reshape(1, 1, ctx, ctx))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._safe_softmax.default(scores + self.mask, -1)


class _GatheredMaskModel(nn.Module):
    """add(scores, mask[pos]) -> _safe_softmax: the MEC mask-gathered-by-pos form."""

    def __init__(self, ctx: int) -> None:
        super().__init__()
        self.register_buffer("mask", _causal_mask(ctx))

    def forward(self, scores: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._safe_softmax.default(scores + self.mask[pos], -1)


class _NoMaskModel(nn.Module):
    """Plain softmax with no preceding mask add -- the pass must not fire."""

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._safe_softmax.default(scores, -1)


class _NonZeroKeepMaskModel(nn.Module):
    """add(scores, mask) -> _safe_softmax where the mask's keep value is 5.0, not 0.

    Not the canonical {0, -inf} additive mask, so the pass declines to fuse it."""

    def __init__(self, ctx: int) -> None:
        super().__init__()
        mask = torch.full((ctx, ctx), 5.0)
        mask.masked_fill_(
            torch.triu(torch.ones(ctx, ctx), diagonal=1).bool(), float("-inf")
        )
        self.register_buffer("mask", mask.reshape(1, 1, ctx, ctx))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._safe_softmax.default(scores + self.mask, -1)


class _NonConstAddModel(nn.Module):
    """softmax over add(scores, bias) where bias is an activation, not a constant."""

    def forward(self, scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._safe_softmax.default(scores + bias, -1)


class _AlphaMaskModel(nn.Module):
    """add(scores, mask, alpha=2) -> _safe_softmax: a scaled add.

    add.Tensor computes self + alpha*other; only alpha=1 combines the operands
    as-is, so the recompose must decline a non-default alpha."""

    def __init__(self, ctx: int) -> None:
        super().__init__()
        self.register_buffer("mask", _causal_mask(ctx).reshape(1, 1, ctx, ctx))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._safe_softmax.default(
            torch.ops.aten.add.Tensor(scores, self.mask, alpha=2), -1
        )


def _count(ep: ExportedProgram, target: object) -> int:
    return len(ep.graph_module.graph.find_nodes(op="call_function", target=target))


def _only(ep: ExportedProgram, target: object) -> torch.fx.Node:
    nodes = ep.graph_module.graph.find_nodes(op="call_function", target=target)
    assert len(nodes) == 1
    return nodes[0]


def _lifted_fqns(ep: ExportedProgram) -> set[str]:
    """Fully-qualified names of every lifted constant (params/buffers in state_dict
    plus lifted tensor constants)."""
    return set(ep.state_dict.keys()) | set(ep.constants.keys())


class FuseAddSoftmaxIntoMaskedSoftmaxTest(unittest.TestCase):
    _CTX = 4

    def _build(self, gathered: bool) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)
        if gathered:
            model = _GatheredMaskModel(self._CTX)
            # scores: [B, H, seq, ctx]; pos selects `seq` rows of the mask.
            inputs = (torch.randn(1, 2, 3, self._CTX), torch.tensor([0, 1, 2]))
        else:
            model = _DirectMaskModel(self._CTX)
            inputs = (torch.randn(1, 2, self._CTX, self._CTX),)
        return model, inputs

    @parameterized.expand([("direct", False), ("gathered", True)])
    def test_fuses_to_masked_softmax(self, _name: str, gathered: bool) -> None:
        model, inputs = self._build(gathered)
        ep = trace(model, inputs)

        result = FuseAddSoftmaxIntoMaskedSoftmax().call(ep)
        self.assertTrue(result.modified)
        ep = result.exported_program

        # The add + safe_softmax pair is replaced by a single _masked_softmax.
        self.assertEqual(_count(ep, _MASKED_SOFTMAX), 1)
        self.assertEqual(_count(ep, _SAFE_SOFTMAX), 0)
        self.assertEqual(_count(ep, _ADD), 0)

        masked_softmax = _only(ep, _MASKED_SOFTMAX)
        _scores, mask, dim, mask_type = masked_softmax.args
        mask = cast(fx.Node, mask)
        # The mask operand is a bool tensor (never quantized), broadcast to the
        # full scores shape for mask_type 2.
        self.assertEqual(mask.meta["val"].dtype, torch.bool)
        self.assertEqual(
            tuple(mask.meta["val"].shape),
            (1, 2, 3, self._CTX) if gathered else (1, 2, self._CTX, self._CTX),
        )
        self.assertEqual(dim, -1)
        self.assertEqual(mask_type, _MASK_TYPE_FULL)

    def test_gathered_mask_index_is_rewired_to_bool(self) -> None:
        """The runtime index gather is preserved but now reads a bool buffer."""
        model, inputs = self._build(gathered=True)
        ep = trace(model, inputs)

        FuseAddSoftmaxIntoMaskedSoftmax().call(ep)

        index_nodes = ep.graph_module.graph.find_nodes(
            op="call_function", target=_INDEX
        )
        self.assertEqual(len(index_nodes), 1)
        self.assertEqual(index_nodes[0].meta["val"].dtype, torch.bool)

    @parameterized.expand([("direct", False), ("gathered", True)])
    def test_numerically_equivalent(self, _name: str, gathered: bool) -> None:
        model, inputs = self._build(gathered)
        with torch.no_grad():
            ref = model(*inputs)

        ep = trace(model, inputs)
        FuseAddSoftmaxIntoMaskedSoftmax().call(ep)
        with torch.no_grad():
            got = ep.module()(*inputs)

        # The mask uses -inf, so additive softmax and exact masking agree exactly
        # (every causal row has at least one unmasked position -> no NaN guard).
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)

    def test_source_fn_stack_is_copied(self) -> None:
        """The quantizer's matcher skips nodes without source_fn_stack, so the
        recompose must carry it from the softmax onto the new op."""
        model, inputs = self._build(gathered=False)
        ep = trace(model, inputs)
        sentinel = [("attn_softmax", torch.nn.functional.softmax)]
        _only(ep, _SAFE_SOFTMAX).meta["source_fn_stack"] = sentinel

        FuseAddSoftmaxIntoMaskedSoftmax().call(ep)

        self.assertEqual(
            _only(ep, _MASKED_SOFTMAX).meta.get("source_fn_stack"), sentinel
        )

    def test_no_mask_is_noop(self) -> None:
        torch.manual_seed(0)
        ep = trace(_NoMaskModel(), (torch.randn(1, 2, self._CTX, self._CTX),))

        result = FuseAddSoftmaxIntoMaskedSoftmax().call(ep)

        self.assertFalse(result.modified)
        self.assertEqual(_count(result.exported_program, _SAFE_SOFTMAX), 1)
        self.assertEqual(_count(result.exported_program, _MASKED_SOFTMAX), 0)

    def test_non_constant_add_operand_is_noop(self) -> None:
        """An add whose operands are both activations is not a mask add."""
        torch.manual_seed(0)
        scores = torch.randn(1, 2, self._CTX, self._CTX)
        bias = torch.randn(1, 2, self._CTX, self._CTX)
        ep = trace(_NonConstAddModel(), (scores, bias))

        result = FuseAddSoftmaxIntoMaskedSoftmax().call(ep)

        self.assertFalse(result.modified)
        self.assertEqual(_count(result.exported_program, _MASKED_SOFTMAX), 0)
        self.assertEqual(_count(result.exported_program, _SAFE_SOFTMAX), 1)

    def test_nonzero_keep_value_is_noop(self) -> None:
        """A binary mask whose keep positions are nonzero is not the canonical
        {0, -inf} form, so the pass conservatively declines to fuse it."""
        torch.manual_seed(0)
        ep = trace(
            _NonZeroKeepMaskModel(self._CTX),
            (torch.randn(1, 2, self._CTX, self._CTX),),
        )

        result = FuseAddSoftmaxIntoMaskedSoftmax().call(ep)

        self.assertFalse(result.modified)
        self.assertEqual(_count(result.exported_program, _MASKED_SOFTMAX), 0)
        self.assertEqual(_count(result.exported_program, _SAFE_SOFTMAX), 1)

    def test_alpha_not_one_is_noop(self) -> None:
        """add(scores, mask, alpha != 1) scales an operand, so it is not fusible."""
        torch.manual_seed(0)
        ep = trace(
            _AlphaMaskModel(self._CTX),
            (torch.randn(1, 2, self._CTX, self._CTX),),
        )

        result = FuseAddSoftmaxIntoMaskedSoftmax().call(ep)

        self.assertFalse(result.modified)
        self.assertEqual(_count(result.exported_program, _MASKED_SOFTMAX), 0)
        self.assertEqual(_count(result.exported_program, _SAFE_SOFTMAX), 1)

    @parameterized.expand([("direct", False), ("gathered", True)])
    def test_additive_mask_constant_is_removed(
        self, _name: str, gathered: bool
    ) -> None:
        """After fusion the additive mask buffer is gone from the program -- its
        lifted constant, placeholder, and input spec are all removed (constant_prop
        drops the now-dead constant) -- leaving the graph signature consistent."""
        model, inputs = self._build(gathered)
        ep = trace(model, inputs)
        # Sanity: the additive "mask" buffer is a lifted constant before fusion.
        self.assertIn("mask", _lifted_fqns(ep))

        ep = FuseAddSoftmaxIntoMaskedSoftmax().call(ep).exported_program

        # The additive mask is gone from state_dict/constants...
        self.assertNotIn("mask", _lifted_fqns(ep))
        # ...and the signature stays consistent: one input spec per placeholder.
        self.assertEqual(
            len(ep.graph_signature.input_specs),
            len(ep.graph_module.graph.find_nodes(op="placeholder")),
        )


class IsAdditiveMaskTest(unittest.TestCase):
    """Direct coverage of the additive-mask predicate: only the canonical
    {0, very-negative} two-value float tensor qualifies."""

    def test_canonical_zero_and_neg_inf(self) -> None:
        t = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]])
        self.assertTrue(_is_additive_mask(t))

    def test_canonical_zero_and_finite_sentinel(self) -> None:
        t = torch.tensor([[0.0, -1e4], [0.0, 0.0]])
        self.assertTrue(_is_additive_mask(t))

    def test_nonzero_keep_value_rejected(self) -> None:
        t = torch.tensor([[5.0, float("-inf")], [5.0, 5.0]])
        self.assertFalse(_is_additive_mask(t))

    def test_single_all_masked_value_rejected(self) -> None:
        # A uniform very-negative tensor would softmax to uniform weights, not
        # zeros -- it must not be treated as a mask.
        t = torch.full((2, 2), -1e4)
        self.assertFalse(_is_additive_mask(t))

    def test_negative_value_above_threshold_rejected(self) -> None:
        # The non-zero value is negative but not a masking sentinel.
        t = torch.tensor([[0.0, -5.0], [0.0, 0.0]])
        self.assertFalse(_is_additive_mask(t))

    def test_integer_tensor_rejected(self) -> None:
        t = torch.tensor([[0, -10000], [0, 0]], dtype=torch.int64)
        self.assertFalse(_is_additive_mask(t))


class HasFullyMaskedRowTest(unittest.TestCase):
    """Direct coverage of the fully-masked-row guard. Inputs are canonical additive
    masks ({0, sentinel}), which is what _has_fully_masked_row assumes (it runs
    only after _is_additive_mask)."""

    def test_causal_mask_has_no_fully_masked_row(self) -> None:
        # 0 on/below the diagonal, -inf above: every row keeps its diagonal.
        self.assertFalse(_has_fully_masked_row(_causal_mask(4), dim=-1, scores_rank=2))

    def test_all_masked_row_is_detected(self) -> None:
        mask = _causal_mask(4)
        mask[2] = float("-inf")  # query row 2 attends to nothing
        self.assertTrue(_has_fully_masked_row(mask, dim=-1, scores_rank=2))

    def test_finite_sentinel_row_is_detected(self) -> None:
        # The sentinel need not be -inf; -1e4 is the min and marks masked positions.
        mask = torch.zeros(3, 3)
        mask[1] = -1e4
        self.assertTrue(_has_fully_masked_row(mask, dim=-1, scores_rank=2))

    def test_lower_rank_mask_maps_dim_from_the_right(self) -> None:
        # Gathered form: a [ctx, ctx] buffer checked against rank-4 scores -- the
        # softmax dim -1 still lines up with the mask's last axis.
        mask = _causal_mask(4)
        self.assertFalse(_has_fully_masked_row(mask, dim=-1, scores_rank=4))
        mask[0] = float("-inf")
        self.assertTrue(_has_fully_masked_row(mask, dim=-1, scores_rank=4))

    def test_positive_dim_is_normalized(self) -> None:
        # dim=3 on rank-4 scores is the same axis as dim=-1.
        mask = _causal_mask(4).reshape(1, 1, 4, 4)
        self.assertFalse(_has_fully_masked_row(mask, dim=3, scores_rank=4))

    def test_mask_not_spanning_softmax_dim_is_conservative(self) -> None:
        # A 1-D key mask softmaxed over a dim it doesn't span: the mask broadcasts
        # along that dim, so any masked element implies a fully-masked row.
        mask = torch.tensor([0.0, 0.0, -1e4])
        self.assertTrue(_has_fully_masked_row(mask, dim=-2, scores_rank=3))
