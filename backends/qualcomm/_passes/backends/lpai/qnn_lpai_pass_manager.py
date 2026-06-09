# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm._passes import (
    DecomposeHardsigmoid,
    DecomposeReciprocal,
    FoldQDQ,
    RemoveRedundancy,
)
from executorch.backends.qualcomm._passes.backends.lpai.fold_qdq import LpaiFoldQDQ
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager


class QnnLpaiPassManager(QnnPassManager):
    """
    Pass manager for the LPAI backend.

    Extends QnnPassManager with LPAI-specific graph transformations.
    """

    @classmethod
    def get_default_pass_activations(cls):
        pass_activations = super().get_default_pass_activations()
        pass_activations = [
            (LpaiFoldQDQ if p is FoldQDQ else p, act) for p, act in pass_activations
        ]
        # Hardsigmoid and Reciprocal no longer appear at to_edge stage as it is decomposed in the export/annotation pipeline.
        # The current change is intended to proactively prepare for the upcoming deprecation of the export pipeline.
        pass_activations.extend(
            [
                (DecomposeHardsigmoid, True),
                (DecomposeReciprocal, True),
            ]
        )
        return pass_activations

    @classmethod
    def get_passes_dependency_for_capture_program(cls):
        deps = super().get_passes_dependency_for_capture_program()
        # Replace FoldQDQ with LpaiFoldQDQ in the dependency table
        if FoldQDQ in deps:
            deps[LpaiFoldQDQ] = deps.pop(FoldQDQ)
        for key in deps:
            deps[key] = [LpaiFoldQDQ if v is FoldQDQ else v for v in deps[key]]
        # Hardsigmoid and Reciprocal no longer appear at to_edge stage as it is decomposed in the export/annotation pipeline.
        # The current change is intended to proactively prepare for the upcoming deprecation of the export pipeline.
        deps.update(
            {
                DecomposeHardsigmoid: [RemoveRedundancy],
                DecomposeReciprocal: [RemoveRedundancy],
            }
        )
        return deps

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
        passes = super().get_preprocess_passes(use_mha2sha)
        return [LpaiFoldQDQ if p is FoldQDQ else p for p in passes]
