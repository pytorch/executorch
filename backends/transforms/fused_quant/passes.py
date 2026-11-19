# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.exir.dialects.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.transforms.advance_quantize_op_above_def_chain import (
    AdvanceQuantizeOpAboveDefChainPass,
)
from executorch.backends.transforms.advance_quantize_op_above_def_in_branch import (
    AdvanceQuantizeOpAboveDefInBranchPass,
)
from executorch.backends.transforms.canonicalize_view_copy_permute_pass import (
    CanonicalizeViewCopyPermutePass,
)
from executorch.backends.transforms.fuse_cascaded_transpose_or_permute_ops import (
    FuseCascadedTransposeOrPermuteOps,
)
from executorch.backends.transforms.fuse_cascaded_view_ops import FuseCascadedViewOps
from executorch.backends.transforms.fuse_quant_dequant_to_requantize import (
    FuseQuantDequantToRequantizePass,
)
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    has_only_per_tensor_qparams,
)
from executorch.backends.transforms.fused_quant.optimization_passes.constant_fold import (
    ConstantFold,
)
from executorch.backends.transforms.fused_quant.optimization_passes.fold_relu_into_producer import (
    FoldReluIntoProducer,
)
from executorch.backends.transforms.fused_quant.optimization_passes.fold_scalar_mul import (
    FoldScalarMul,
)
from executorch.backends.transforms.fused_quant.optimization_passes.fuse_add_into_linear import (
    FuseAddIntoLinear,
)
from executorch.backends.transforms.fused_quant.optimization_passes.fuse_mul_into_linear import (
    FuseMulIntoLinear,
)
from executorch.backends.transforms.fused_quant.optimization_passes.hoist_activations import (
    HoistActivations,
)
from executorch.backends.transforms.fused_quant.optimization_passes.prequantize_embedding import (
    PrequantizeEmbedding,
)
from executorch.backends.transforms.fused_quant.optimization_passes.quant_absorption import (
    QuantAbsorptionPass,
)
from executorch.backends.transforms.fused_quant.optimization_passes.replace_dequant_quant_with_requantize import (
    ReplaceDequantQuantWithRequantize,
)
from executorch.backends.transforms.fused_quant.optimization_passes.replace_sub_constant_with_add import (
    ReplaceSubConstantWithAdd,
)
from executorch.backends.transforms.fused_quant.optimization_passes.sink_constant_cat import (
    SinkConstantCat,
)
from executorch.backends.transforms.fused_quant.optimization_passes.split_linear_at_slices import (
    SplitLinearAtSlices,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_last import (
    ToChannelsLast,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_convolution import (
    ToConvolution,
)
from executorch.backends.transforms.fused_quant.pass_base import IterativePassGroup
from executorch.backends.transforms.merge_split_concat_chain import (
    MergeSplitConcatChainPass,
)
from executorch.backends.transforms.move_permute_after_concat import (
    MovePermuteAfterConcat,
)
from executorch.backends.transforms.move_slice_before_permute import (
    MoveSliceBeforePermutePass,
)
from executorch.backends.transforms.move_slice_before_view import (
    MoveSliceBeforeViewPass,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.postpone_dequantize_op_below_use_chain import (
    PostponeDequantizeOpBelowUseChainPass,
)
from executorch.backends.transforms.propagate_slice import PropagateSlice
from executorch.backends.transforms.remove_alias_copy_op import RemoveAliasCopyOpPass
from executorch.backends.transforms.remove_branched_quant_dequant import (
    RemoveBranchedQuantDequant,
)
from executorch.backends.transforms.remove_cat_from_slice_copy import (
    RemoveCatFromSliceCopyPass,
)
from executorch.backends.transforms.remove_clone_ops_transform_imported import (
    RemoveCloneOpsTransformImported,
)
from executorch.backends.transforms.remove_nop_as_strided_copy_op import (
    RemoveNopAsStridedCopyOpPass,
)
from executorch.backends.transforms.remove_nop_expand_op import RemoveNopExpandOpPass
from executorch.backends.transforms.remove_nop_slice_or_view_op import (
    RemoveNopSliceOrViewOpPass,
)
from executorch.backends.transforms.remove_permute_before_mean import (
    RemovePermuteBeforeMeanPass,
)
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.backends.transforms.remove_zero_sized_cat_args import (
    RemoveZeroSizedCatArgsPass,
)
from executorch.backends.transforms.replace_nop_transpose_or_permute_with_view import (
    ReplaceNopTransposeOrPermuteWithViewPass,
)
from executorch.backends.transforms.replace_select_with_view_op import (
    ReplaceSelectWithViewOpPass,
)
from executorch.backends.transforms.replace_split_with_slice import (
    ReplaceSplitWithSlicePass,
)
from executorch.backends.transforms.replace_squeeze_unsqueeze_with_view import (
    ReplaceSqueezeAndUnsqueezeWithViewPass,
)
from executorch.backends.transforms.split_dequantized_cat import SplitDequantizedCatPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_manager import PassType
from torch.fx.passes.dialect.common.cse_pass import CSEPass, get_CSE_banned_ops

# fused_quant elementwise ops (add / mul in both their Tensor and Scalar
# overloads, and the unary activations) that RemovePermutesAroundElementwiseOps
# should treat as permutable, so a permutation from ToChannelsLast propagates
# through them and cancels at the boundary permutes -- running the whole region
# in NHWC. See the pass for how the lifted scale / zero_point operands are
# handled per qparam flavor.
FUSED_QUANT_ELEMENTWISE: set[EdgeOpOverload] = {
    exir_ops.edge.fused_quant.add.default,
    exir_ops.edge.fused_quant.add.Scalar,
    exir_ops.edge.fused_quant.sub.default,
    exir_ops.edge.fused_quant.sub.Scalar,
    exir_ops.edge.fused_quant.mul.default,
    exir_ops.edge.fused_quant.mul.Scalar,
    exir_ops.edge.fused_quant.relu.default,
    exir_ops.edge.fused_quant.hardswish.default,
    exir_ops.edge.fused_quant.sigmoid.default,
    exir_ops.edge.fused_quant.tanh.default,
    exir_ops.edge.fused_quant.hard_tanh.default,
    exir_ops.edge.fused_quant.silu.default,
    exir_ops.edge.fused_quant.hardsigmoid.default,
    exir_ops.edge.fused_quant.gelu.default,
}

# Hardcoded max number of iterations for the optimization group.
_OPTIMIZATION_STEPS = 5


def _can_fuse_last_dim_softmax_through_view(
    softmax: torch.fx.Node,
    final_view: torch.fx.Node,
) -> bool:
    """Whether folding ``final_view`` across ``softmax`` preserves its groups."""

    # Can be overly restrictive, but this is the typical case.
    if get_arg(softmax, "dim", int) != -1:
        return False

    inp = get_arg(softmax, "input", torch.fx.Node)
    input_shape = inp.meta["val"].shape
    final_shape = final_view.meta["val"].shape
    return len(final_shape) > 0 and input_shape[-1] == final_shape[-1]


def _can_fuse_square_linear_through_view(
    linear: torch.fx.Node,
    final_view: torch.fx.Node,
) -> bool:
    """Whether folding ``final_view`` across ``linear`` preserves its shape.

    The fold changes the rank of the linear's input and output, so their qparams
    must be per-tensor.
    """

    for prefix in ("inp", "out"):
        qparams = get_qparams_from_node(linear, prefix)
        if qparams is not None and not qparams.is_per_tensor():
            return False

    inp = get_arg(linear, "inp", torch.fx.Node)
    input_shape = inp.meta["val"].shape
    output_shape = linear.meta["val"].shape
    final_shape = final_view.meta["val"].shape
    return (
        len(final_shape) > 0
        and input_shape[-1] == output_shape[-1]
        and output_shape[-1] == final_shape[-1]
    )


def get_optimization_passes(
    *,
    absorb_quant_with_fork: bool = False,
    absorb_quant_into_inputs: bool = False,
    symmetric_quant_input_ops: frozenset[EdgeOpOverload] = frozenset(),
    extra_quantizable_ops: dict[EdgeOpOverload, tuple[int, ...]] | None = None,
    channels_last: bool = False,
    fold_relu_into_producer: bool = False,
    fold_into_linear: bool = False,
    prequantize_embedding: bool = False,
    split_linear_at_slices: bool = False,
) -> IterativePassGroup:
    """The rewrites applied repeatedly to the fused_quant edge graph.

    Returns the group rather than running it. The passes feed each other -- a
    layout change exposes a fusion, which exposes a dead transpose -- so they are
    re-run until no pass reports a change, capped at ``_OPTIMIZATION_STEPS``
    iterations. The cap matters: the group is not guaranteed to converge, and a
    graph that is still changing on the last iteration is left as-is, with a
    warning.

    ``absorb_quant_with_fork`` and ``absorb_quant_into_inputs`` turn on quant
    absorption's two modes that are not always more accurate; see
    ``QuantAbsorptionPass`` for why they are off by default.

    ``symmetric_quant_input_ops`` are ops whose inputs must keep a zero point of
    0, because the backend kernel ignores input zero points; quant absorption
    leaves their inputs alone when absorbing would give them a nonzero one.

    The rewrites that make a choice for the backend are off by default:

    - ``channels_last``: ``ToChannelsLast`` runs convolutions and pools in NHWC,
      which a backend with its own layout pass has to undo.
    - ``fold_relu_into_producer``: ``FoldReluIntoProducer`` has the conv or linear
      before a relu quantize straight to the relu's output, dropping the rounding
      between the two.
    - ``fold_into_linear``: ``FuseAddIntoLinear`` and ``FuseMulIntoLinear`` fold a
      scalar add or a constant multiply into the linear or convolution before it,
      dropping the rounding between the two.
    - ``prequantize_embedding``: ``PrequantizeEmbedding`` turns an embedding with
      a quantized table into an int8 gather before any backend sees it.
    - ``split_linear_at_slices``: ``SplitLinearAtSlices`` splits a linear whose
      output is sliced, such as a fused QKV projection, into one linear per slice.

    Without them the group only canonicalizes and simplifies the graph, keeping
    its layout. Output-side quant absorption stays on: it rounds once where the
    q/dq graph rounds twice, which is never less accurate.
    """
    return IterativePassGroup(
        [
            ToConvolution(),
            ReplaceSubConstantWithAdd(),
            QuantAbsorptionPass(
                absorb_with_fork=absorb_quant_with_fork,
                absorb_into_inputs=absorb_quant_into_inputs,
                symmetric_quant_input_ops=symmetric_quant_input_ops,
            ),
            *([ToChannelsLast()] if channels_last else []),
            HoistActivations(),
            *([FoldReluIntoProducer()] if fold_relu_into_producer else []),
            CSEPass(banned_ops=get_CSE_banned_ops()),
            *([FuseAddIntoLinear(), FuseMulIntoLinear()] if fold_into_linear else []),
            # After QuantAbsorptionPass and FuseMulIntoLinear: absorption makes an
            # affine producer and its multiply adjacent, so FuseMulIntoLinear can
            # fold the constant into the weight scale. FoldScalarMul handles the
            # multiplies left over, leaving them as a scale ratio for absorption.
            FoldScalarMul(),
            RemoveAliasCopyOpPass(),
            RemoveCloneOpsTransformImported(),
            RemoveNopAsStridedCopyOpPass(),
            RemoveNopExpandOpPass(),
            RemoveNopSliceOrViewOpPass(),
            FuseCascadedTransposeOrPermuteOps(),
            FuseCascadedViewOps(),
            # Canonicalize rank-changing shape ops to view_copy so the permute
            # pass below sees a single representation.
            ReplaceSqueezeAndUnsqueezeWithViewPass(),
            FuseViewCopyTransform(
                extra_fusible_ops={
                    exir_ops.edge.aten._softmax.default,
                    exir_ops.edge.fused_quant.linear.default,
                },
                target_filters={
                    exir_ops.edge.aten._softmax.default: _can_fuse_last_dim_softmax_through_view,
                    exir_ops.edge.fused_quant.linear.default: _can_fuse_square_linear_through_view,
                },
            ),
            MergeSplitConcatChainPass(),
            ReplaceSplitWithSlicePass(),
            ReplaceSelectWithViewOpPass(),
            PostponeDequantizeOpBelowUseChainPass(),
            RemovePermuteBeforeMeanPass(),
            RemovePermutesAroundElementwiseOps(
                extra_permutable_ops=FUSED_QUANT_ELEMENTWISE,
                target_filters={
                    target: has_only_per_tensor_qparams
                    for target in FUSED_QUANT_ELEMENTWISE
                },
            ),
            MovePermuteAfterConcat(),
            ReplaceNopTransposeOrPermuteWithViewPass(),
            SplitDequantizedCatPass(),
            AdvanceQuantizeOpAboveDefChainPass(
                extra_quantizable_ops=extra_quantizable_ops
            ),
            PropagateSlice(
                additional_binary_targets=[
                    exir_ops.edge.fused_quant.add.default,
                    exir_ops.edge.fused_quant.mul.default,
                ],
                target_filters={
                    exir_ops.edge.fused_quant.add.default: has_only_per_tensor_qparams,
                    exir_ops.edge.fused_quant.mul.default: has_only_per_tensor_qparams,
                },
            ),
            AdvanceQuantizeOpAboveDefInBranchPass(),
            FuseQuantDequantToRequantizePass(allow_requantize=False),
            RemoveBranchedQuantDequant(),
            *([PrequantizeEmbedding()] if prequantize_embedding else []),
            SinkConstantCat(),
            *([SplitLinearAtSlices()] if split_linear_at_slices else []),
            RemoveCatFromSliceCopyPass(),
            RemoveZeroSizedCatArgsPass(),
            MoveSliceBeforePermutePass(),
            MoveSliceBeforeViewPass(),
            CanonicalizeViewCopyPermutePass(),
        ],
        _OPTIMIZATION_STEPS,
    )


def get_fused_quant_passes(
    *,
    absorb_quant_with_fork: bool = False,
    absorb_quant_into_inputs: bool = False,
    symmetric_quant_input_ops: frozenset[EdgeOpOverload] = frozenset(),
    extra_quantizable_ops: dict[EdgeOpOverload, tuple[int, ...]] | None = None,
    channels_last: bool = False,
    fold_relu_into_producer: bool = False,
    fold_into_linear: bool = False,
    prequantize_embedding: bool = False,
    split_linear_at_slices: bool = False,
) -> list[PassType]:
    """The backend-neutral pipeline that produces the optimized fused_quant graph.

    Runs on a graph that already contains fused_quant ops in edge dialect (fusion
    happens before ``to_edge``) and stops before coloring, so the result is
    hardware-independent. A backend appends its own coloring, lowering, and
    partitioning passes after these, and turns on the rewrites of
    ``get_optimization_passes`` it wants.
    """
    return [
        # Fold pure-constant subgraphs before the optimization passes. Most
        # importantly this bakes the aten.view_copy that fusion inserts to
        # reshape a per-channel weight scale/zero_point to full rank: without
        # it those qparams reach the optimization passes as a non-constant
        # view_copy, so passes that read them via get_constant (e.g.
        # SplitLinearAtSlices) see None and skip the rewrite.
        ConstantFold(),
        get_optimization_passes(
            absorb_quant_with_fork=absorb_quant_with_fork,
            absorb_quant_into_inputs=absorb_quant_into_inputs,
            symmetric_quant_input_ops=symmetric_quant_input_ops,
            extra_quantizable_ops=extra_quantizable_ops,
            channels_last=channels_last,
            fold_relu_into_producer=fold_relu_into_producer,
            fold_into_linear=fold_into_linear,
            prequantize_embedding=prequantize_embedding,
            split_linear_at_slices=split_linear_at_slices,
        ),
        ConstantFold(),
        ReplaceDequantQuantWithRequantize(),
    ]
