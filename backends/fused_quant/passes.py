# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import executorch.backends.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.cadence.aot.fuse_ops import (
    FuseCascadedTransposeOrPermuteOps,
    FuseCascadedViewOps,
    FuseQuantDequantToRequantizePass,
)
from executorch.backends.cadence.aot.remove_ops import (
    RemoveAliasCopyOpPass,
    RemoveBranchedQuantDequant,
    RemoveCatFromSliceCopyPass,
    RemoveCloneOpsTransformImported,
    RemoveNopExpandOpPass,
    RemoveNopSliceOrViewOpPass,
    RemovePermuteBeforeMeanPass,
    RemoveZeroSizedCatArgsPass,
)
from executorch.backends.cadence.aot.reorder_ops import (
    AdvanceQuantizeOpAboveDefChainPass,
    AdvanceQuantizeOpAboveDefInBranchPass,
    MovePermuteAfterConcat,
    MoveSliceBeforePermutePass,
    MoveSliceBeforeViewPass,
    PostponeDequantizeOpBelowUseChainPass,
    PropagateSlice,
    SplitDequantizedCatPass,
)
from executorch.backends.cadence.aot.replace_ops import (
    ReplaceNopTransposeOrPermuteWithViewPass,
    ReplaceSelectWithViewOpPass,
    ReplaceSplitWithSlicePass,
)
from executorch.backends.fused_quant.graph_utils import (
    get_qparams_from_node,
    split_fused_arg_names,
)
from executorch.backends.fused_quant.optimization_passes.constant_fold import (
    ConstantFold,
)
from executorch.backends.fused_quant.optimization_passes.fuse_add_into_linear import (
    FuseAddIntoLinear,
)
from executorch.backends.fused_quant.optimization_passes.fuse_mul_into_linear import (
    FuseMulIntoLinear,
)
from executorch.backends.fused_quant.optimization_passes.hoist_activations import (
    HoistActivations,
)
from executorch.backends.fused_quant.optimization_passes.prequantize_embedding import (
    PrequantizeEmbedding,
)
from executorch.backends.fused_quant.optimization_passes.quant_absorption import (
    QuantAbsorptionPass,
)
from executorch.backends.fused_quant.optimization_passes.replace_dequant_quant_with_requantize import (
    ReplaceDequantQuantWithRequantize,
)
from executorch.backends.fused_quant.optimization_passes.replace_scalar_mul_with_dequant_quant import (
    ReplaceScalarMulWithDequantQuant,
)
from executorch.backends.fused_quant.optimization_passes.sink_constant_cat import (
    SinkConstantCat,
)
from executorch.backends.fused_quant.optimization_passes.split_linear_at_slices import (
    SplitLinearAtSlices,
)
from executorch.backends.fused_quant.optimization_passes.to_channels_last import (
    ToChannelsLast,
)
from executorch.backends.fused_quant.optimization_passes.to_convolution import (
    ToConvolution,
)
from executorch.backends.fused_quant.pass_base import IterativePassGroup
from executorch.backends.transforms.merge_split_concat_chain import (
    MergeSplitConcatChainPass,
)
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.backends.transforms.replace_squeeze_unsqueeze_with_view import (
    ReplaceSqueezeAndUnsqueezeWithViewPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_manager import PassType
from torch.fx.passes.dialect.common.cse_pass import CSEPass

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


def _has_only_per_tensor_qparams(node: torch.fx.Node) -> bool:
    """Whether every qparam block the op carries is per-tensor.

    Which blocks exist depends on the op: unary fused ops have no ``other``.
    """
    node_target = cast(EdgeOpOverload, node.target)
    input_names, _ = split_fused_arg_names(node_target)
    for prefix in (*input_names, "out"):
        qparams = get_qparams_from_node(node, prefix)
        if qparams is not None and not qparams.is_per_tensor():
            return False
    return True


def get_optimization_passes(
    *,
    absorb_quant_with_fork: bool = False,
    extra_quantizable_ops: dict[EdgeOpOverload, tuple[int, ...]] | None = None,
) -> IterativePassGroup:
    """The rewrites applied repeatedly to the fused_quant edge graph.

    Returns the group rather than running it. The passes feed each other -- a
    layout change exposes a fusion, which exposes a dead transpose -- so they are
    re-run until no pass reports a change, capped at ``_OPTIMIZATION_STEPS``
    iterations. The cap matters: the group is not guaranteed to converge, and a
    graph that is still changing on the last iteration is simply left as-is.
    """
    return IterativePassGroup(
        [
            ToConvolution(),
            ReplaceScalarMulWithDequantQuant(),
            QuantAbsorptionPass(absorb_with_fork=absorb_quant_with_fork),
            ToChannelsLast(),
            HoistActivations(),
            CSEPass(),
            FuseAddIntoLinear(),
            FuseMulIntoLinear(),
            RemoveAliasCopyOpPass(),
            RemoveCloneOpsTransformImported(),
            RemoveNopExpandOpPass(),
            RemoveNopSliceOrViewOpPass(),
            FuseCascadedTransposeOrPermuteOps(),
            FuseCascadedViewOps(),
            # Canonicalize rank-changing shape ops to view_copy so the permute
            # pass below sees a single representation.
            ReplaceSqueezeAndUnsqueezeWithViewPass(),
            MergeSplitConcatChainPass(),
            ReplaceSplitWithSlicePass(),
            ReplaceSelectWithViewOpPass(),
            PostponeDequantizeOpBelowUseChainPass(),
            RemovePermuteBeforeMeanPass(),
            RemovePermutesAroundElementwiseOps(
                extra_permutable_ops=FUSED_QUANT_ELEMENTWISE,
                target_filters={
                    target: _has_only_per_tensor_qparams
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
                    exir_ops.edge.fused_quant.add.default: _has_only_per_tensor_qparams,
                    exir_ops.edge.fused_quant.mul.default: _has_only_per_tensor_qparams,
                },
            ),
            AdvanceQuantizeOpAboveDefInBranchPass(),
            FuseQuantDequantToRequantizePass(allow_requantize=False),
            RemoveBranchedQuantDequant(),
            PrequantizeEmbedding(),
            SinkConstantCat(),
            SplitLinearAtSlices(),
            RemoveCatFromSliceCopyPass(),
            RemoveZeroSizedCatArgsPass(),
            MoveSliceBeforePermutePass(),
            MoveSliceBeforeViewPass(),
        ],
        _OPTIMIZATION_STEPS,
    )


def get_fused_quant_passes(
    *,
    absorb_quant_with_fork: bool = False,
    extra_quantizable_ops: dict[EdgeOpOverload, tuple[int, ...]] | None = None,
) -> list[PassType]:
    """The backend-neutral pipeline that produces the optimized fused_quant graph.

    Runs on a graph that already contains fused_quant ops in edge dialect (fusion
    happens before ``to_edge``) and stops before coloring, so the result is
    hardware-independent. A backend appends its own coloring, lowering, and
    partitioning passes after these.
    """
    return [
        # Fold pure-constant subgraphs before the optimization passes. Most
        # importantly this bakes the aten.view_copy that fusion inserts to
        # reshape a per-channel weight scale/zero_point to full rank: without
        # it those qparams reach the optimization passes as a non-constant
        # view_copy, so passes that read them via get_constant (e.g.
        # SplitLinearAtSlices) see None and silently skip -- leaving a
        # sub-linear with a mismatched full-rank scale.
        ConstantFold(),
        get_optimization_passes(
            absorb_quant_with_fork=absorb_quant_with_fork,
            extra_quantizable_ops=extra_quantizable_ops,
        ),
        ConstantFold(),
        ReplaceDequantQuantWithRequantize(),
    ]
