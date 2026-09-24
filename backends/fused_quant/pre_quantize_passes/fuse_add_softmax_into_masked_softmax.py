# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import torch
from executorch.backends.fused_quant.graph_utils import (
    add_constant,
    get_constant,
    get_fqn,
    get_input_kind,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch._ops import OpOverload
from torch.export import ExportedProgram

_ADD: OpOverload = torch.ops.aten.add.Tensor
_INDEX: OpOverload = torch.ops.aten.index.Tensor
_EXPAND: OpOverload = torch.ops.aten.expand.default
_MASKED_SOFTMAX: OpOverload = torch.ops.aten._masked_softmax.default
# _masked_softmax mask_type 2 is the generic "mask matches input shape" form. The
# 2D forms (0 = square LxL attention, 1 = BxL padding) don't cover an [L_q, L_k]
# mask with a KV cache (L_q != L_k), so we always broadcast to the full input.
_MASK_TYPE_FULL: int = 2
_SOFTMAX_TARGETS: tuple[OpOverload, ...] = (
    torch.ops.aten._safe_softmax.default,
    torch.ops.aten.softmax.int,
    torch.ops.aten._softmax.default,
)

# Sentinel mask values (-1e4, -inf, finfo.min, and their bf16 roundings) sit far
# below this bound; real attention logits never do. So a binary constant holding
# a value past this bound is an additive attention mask, not data.
_NEGATIVE_MASK_THRESHOLD: float = -1e3


@dataclass
class _MaskSource:
    """How the additive mask reaches the add.

    ``buffer_node`` / ``buffer_tensor`` are the constant holding the mask.
    ``gather_node`` is the ``index.Tensor`` reading it at runtime (e.g. the MEC
    mask buffer indexed by ``input_pos``), or None when the constant feeds the
    add directly.
    """

    buffer_node: fx.Node
    buffer_tensor: torch.Tensor
    gather_node: fx.Node | None


def _is_additive_mask(t: torch.Tensor) -> bool:
    """True for a canonical additive attention mask: a float tensor with exactly
    two values -- 0 at keep positions and a very-negative sentinel at masked ones.

    The keep value must be exactly 0. A uniform nonzero keep value would in fact
    cancel in the softmax, but demanding 0 keeps detection conservative: it avoids
    matching arbitrary binary tensors that merely happen to hold a very-negative
    entry. It also excludes a single-valued (all-masked) constant, which is *not*
    equivalent to masking -- a uniform -1e4 row softmaxes to uniform weights, not
    zeros.
    """
    if not torch.is_floating_point(t):
        return False
    uniq = torch.unique(t)
    if uniq.numel() != 2:
        return False
    return bool(uniq.max() == 0) and bool(uniq.min() <= _NEGATIVE_MASK_THRESHOLD)


def _has_fully_masked_row(mask: torch.Tensor, dim: int, scores_rank: int) -> bool:
    """True if some softmax row (a slice along ``dim``) is entirely masked.

    Such a row has nothing to normalize over, so the softmax is undefined: eager
    ``aten._masked_softmax`` writes NaN for it, which would poison quant
    observers downstream during calibration. The rewrite bails in that case.
    Canonical causal / sliding-window masks keep the diagonal unmasked and never
    trip this; it guards pathological masks (e.g. an all-padding query row).
    """
    # True = masked (binary {0, sentinel} per _is_additive_mask; min is the sentinel).
    masked = mask == mask.min()
    # The mask broadcasts to the scores from the right, so map the softmax dim to a
    # right-counted (negative) axis that lines up regardless of the mask's rank.
    neg_dim = dim if dim < 0 else dim - scores_rank
    if mask.dim() < -neg_dim:
        # Mask doesn't span the softmax dim (it broadcasts along it): every slice
        # along that dim repeats a single value, so any masked element is a fully
        # masked row.
        return bool(masked.any())
    return bool(masked.all(dim=neg_dim).any())


def _classify_mask(ep: ExportedProgram, operand: fx.Node) -> _MaskSource | None:
    """Resolve an add operand to its additive-mask constant, else None.

    Handles a constant fed directly, or one gathered through a single-use
    ``index.Tensor`` (the MEC mask-buffer-indexed-by-input_pos pattern).
    """
    const = get_constant(ep, operand)
    if const is not None and _is_additive_mask(const):
        return _MaskSource(operand, const, None)

    if operand.target is _INDEX and len(operand.users) == 1:
        buffer_node = operand.args[0]
        if isinstance(buffer_node, fx.Node):
            const = get_constant(ep, buffer_node)
            if const is not None and _is_additive_mask(const):
                return _MaskSource(buffer_node, const, operand)
    return None


def _split_mask_scores(
    ep: ExportedProgram, add: fx.Node
) -> tuple[fx.Node, _MaskSource] | None:
    """Find which add operand is the additive mask.

    Returns (scores_node, mask_source) when exactly one operand resolves to an
    additive mask; None if neither or both qualify (ambiguous).
    """
    lhs, rhs = add.args[0], add.args[1]
    matches: list[tuple[fx.Node, _MaskSource]] = []
    for mask_cand, scores_cand in ((lhs, rhs), (rhs, lhs)):
        if not isinstance(mask_cand, fx.Node) or not isinstance(scores_cand, fx.Node):
            continue
        source = _classify_mask(ep, mask_cand)
        if source is not None:
            matches.append((scores_cand, source))
    if len(matches) != 1:
        return None
    return matches[0]


def _broadcastable_to(src: torch.Size, target: torch.Size) -> bool:
    """True if a tensor of shape src can broadcast/expand to target."""
    try:
        return tuple(torch.broadcast_shapes(src, target)) == tuple(target)
    except RuntimeError:
        return False


def _mask_operand(source: _MaskSource) -> fx.Node:
    """The node feeding the add as the mask (the gather, or the constant itself)."""
    return source.gather_node if source.gather_node is not None else source.buffer_node


def _materialize_bool_mask(
    ep: ExportedProgram,
    source: _MaskSource,
    before: fx.Node,
    cache: dict[str, fx.Node],
) -> fx.Node:
    """Produce the bool mask node _masked_softmax consumes (True = masked).

    Mutates the graph: the additive buffer is converted to bool once per buffer
    (cached). For a gathered mask the existing index op is rewired to read the
    bool buffer so the runtime gather is preserved; otherwise the bool constant
    feeds the op directly. Call only after mask_type has been validated.
    """
    key = source.buffer_node.name
    bool_const = cache.get(key)
    if bool_const is None:
        kind = get_input_kind(ep, source.buffer_node)
        assert kind is not None
        fqn = get_fqn(ep, source.buffer_node)
        assert fqn is not None
        # Binary buffer: the very-negative sentinel is the minimum (see
        # _is_additive_mask). Equality recovers the bool mask exactly.
        bool_buffer = source.buffer_tensor == source.buffer_tensor.min()
        bool_const = add_constant(
            ep, f"{fqn}_bool", bool_buffer.contiguous(), before, kind=kind
        )
        cache[key] = bool_const

    if source.gather_node is None:
        return bool_const

    source.gather_node.update_arg(0, bool_const)
    source.gather_node.meta["val"] = source.gather_node.meta["val"].to(torch.bool)
    return source.gather_node


def _is_plain_softmax(node: fx.Node) -> bool:
    """False for softmaxes that upcast dtype -- _masked_softmax cannot express that."""
    if node.target is torch.ops.aten._softmax.default:
        return not get_arg(node, "half_to_float")
    return get_arg(node, "dtype") is None


class FuseAddSoftmaxIntoMaskedSoftmax(ExportedProgramPassBase):
    """Fuse ``add(scores, additive_mask) -> softmax`` into ``aten._masked_softmax``.

    Runs post-trace / pre-quantize on the ExportedProgram, where the attention
    mask is still an inspectable constant -- either fed directly or gathered
    through an ``index.Tensor`` (the MEC mask-buffer-indexed-by-input_pos
    pattern). The add + softmax pair is collapsed to a single ``_masked_softmax``
    that takes a bool mask (True = masked), so the quantizer can anchor it as one
    op (skipping the mask operand) and the very-negative sentinel never enters the
    quantized domain -- masking becomes exact rather than an additive -1e4
    approximation.

    The additive buffer is converted to a bool buffer once per buffer; a gathered
    mask keeps its runtime gather (rewired to read the bool buffer). Because
    ``_safe_softmax`` is still a single op here, replacing it also drops its
    internal NaN guard. The original additive buffer goes dead and is pruned by
    the subsequent export.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False
        # Bool mask constants keyed by source buffer name, so a buffer used by
        # multiple layers/softmaxes is materialized once.
        bool_mask_cache: dict[str, fx.Node] = {}
        # find_nodes returns a snapshot, so mutation while iterating is safe.
        for target in _SOFTMAX_TARGETS:
            for softmax in graph.find_nodes(op="call_function", target=target):
                modified |= self._try_fuse(exported_program, softmax, bool_mask_cache)

        if modified:
            # The recompose replaced each additive mask with a bool mask, leaving
            # the additive buffer a dead lifted constant. constant_prop_pass erases
            # such dead constants from the graph, state_dict/constants, AND
            # graph_signature (and folds the now-constant bool-mask subgraph),
            # keeping the signature self-consistent without a manual remove_constant.
            # It also DCEs and recompiles.
            exported_program = constant_prop_pass(exported_program)

        return ExportedProgramPassResult(exported_program, modified)

    def _try_fuse(
        self,
        ep: ExportedProgram,
        softmax: fx.Node,
        bool_mask_cache: dict[str, fx.Node],
    ) -> bool:
        """Rewrite add(scores, mask) -> softmax into a single _masked_softmax."""
        if not _is_plain_softmax(softmax):
            return False
        add = softmax.args[0]
        if (
            not isinstance(add, fx.Node)
            or add.target is not _ADD
            or len(add.users) != 1
        ):
            return False

        # add.Tensor computes ``self + alpha * other``; a non-default alpha scales
        # one operand, so scores and mask are not combined as-is and the recompose
        # would be wrong. Only a plain (alpha=1) add is fusible.
        if get_arg(add, "alpha") != 1:
            return False

        split = _split_mask_scores(ep, add)
        if split is None:
            return False
        scores, source = split

        # Validate before mutating: _materialize rewires the gather and adds a
        # constant, so an unusable mask shape must bail first or it would leave a
        # bool gather feeding the still-present add. The mask must broadcast to
        # the scores shape (mask_type 2 requires a full-shape mask).
        scores_shape = scores.meta["val"].shape
        if not _broadcastable_to(_mask_operand(source).meta["val"].shape, scores_shape):
            return False

        # A fully-masked softmax row is undefined and makes eager aten._masked_softmax
        # emit NaN (poisoning quant calibration), so decline the rewrite if the mask
        # has one. Checked on the buffer rather than the gathered operand so a runtime
        # gather can't select a fully-masked row we never inspected.
        dim = get_arg(softmax, "dim", int)
        if _has_fully_masked_row(source.buffer_tensor, dim, len(scores_shape)):
            return False

        graph = ep.graph_module.graph
        mask_node = _materialize_bool_mask(ep, source, softmax, bool_mask_cache)

        # mask_type 2 wants the mask to match the input shape exactly; the gathered
        # [L_q, L_k] mask broadcasts over batch/heads, so expand it to scores shape.
        if tuple(mask_node.meta["val"].shape) != tuple(scores_shape):
            with graph.inserting_before(softmax):
                expanded = graph.call_function(
                    _EXPAND, args=(mask_node, list(scores_shape))
                )
            expanded.meta["val"] = mask_node.meta["val"].expand(scores_shape)
            mask_node = expanded

        with graph.inserting_before(softmax):
            fused = graph.call_function(
                _MASKED_SOFTMAX, args=(scores, mask_node, dim, _MASK_TYPE_FULL)
            )

        fused.meta = softmax.meta

        softmax.replace_all_uses_with(fused)
        graph.erase_node(softmax)
        if not add.users:
            graph.erase_node(add)
        return True
