# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm._passes import (
    ConvertMhaToSha,
    DecomposeHardsigmoid,
    DecomposeReciprocal,
    FoldQDQ,
    FuseConsecutiveCast,
    FuseConsecutiveTranspose,
    InsertRequantize,
    LayoutTransform,
    LpaiPartitionFallbackSupport,
    RemoveRedundancy,
    ResolveDebugHandle,
    TagQuantIO,
)
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager


class QnnLpaiPassManager(QnnPassManager):
    """
    Pass manager for the LPAI backend.

    Extends QnnPassManager with LPAI-specific graph transformations.
    """

    @classmethod
    def get_default_pass_activations(cls):
        pass_activations = super().get_default_pass_activations()
        # Hardsigmoid and Reciprocal no longer appear at to_edge stage as it is decomposed in the export/annotation pipeline.
        # The current change is intended to proactively prepare for the upcoming deprecation of the export pipeline.
        pass_activations.extend(
            [
                (DecomposeHardsigmoid, True),
                (DecomposeReciprocal, True),
                (LpaiPartitionFallbackSupport, True),
            ]
        )
        return pass_activations

    @classmethod
    def get_passes_dependency_for_capture_program(cls):
        deps = super().get_passes_dependency_for_capture_program()
        # Hardsigmoid and Reciprocal no longer appear at to_edge stage as it is decomposed in the export/annotation pipeline.
        # The current change is intended to proactively prepare for the upcoming deprecation of the export pipeline.
        deps.update(
            {
                DecomposeHardsigmoid: [RemoveRedundancy],
                DecomposeReciprocal: [RemoveRedundancy],
                LpaiPartitionFallbackSupport: [TagQuantIO],
                ResolveDebugHandle: [LpaiPartitionFallbackSupport],
            }
        )
        return deps

    def _validate_edge_passes(self) -> None:
        super()._validate_edge_passes()
        assert isinstance(
            self.passes[-2], LpaiPartitionFallbackSupport
        ), "Please ensure LpaiPartitionFallbackSupport is the last edge pass before ResolveDebugHandle."

    @classmethod
    def get_annotation_passes(cls):
        passes = [DecomposeHardsigmoid, DecomposeReciprocal]
        passes.extend(super().get_annotation_passes())
        return passes

    @classmethod
    def get_export_passes(
        cls,
        convert_linear_to_conv2d: bool = False,
    ):
        # Both DecomposeHardSigmoid and DecomposeReciprocal should be placed in the export
        # pipeline, as they rely on LiftConstantScalarOperands to lift the scalar operand.
        passes = [DecomposeHardsigmoid, DecomposeReciprocal]
        passes.extend(super().get_export_passes(convert_linear_to_conv2d))
        return passes

    @classmethod
    def get_preprocess_passes(cls, use_mha2sha=False):
        passes = [
            FoldQDQ,
            ConvertMhaToSha,
            InsertRequantize,
            LayoutTransform,
            FuseConsecutiveCast,
            FuseConsecutiveTranspose,
        ]
        if not use_mha2sha:
            passes.remove(ConvertMhaToSha)
        return passes
