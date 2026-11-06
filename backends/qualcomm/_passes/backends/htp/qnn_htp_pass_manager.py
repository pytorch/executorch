# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from executorch.backends.qualcomm._passes import (
    DecomposeReciprocal,
    I64toI32,
    RecomposeHadamard,
    RemoveRedundancy,
    ResolveDebugHandle,
    TagPositionalSinCos,
)
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager


class QnnHtpPassManager(QnnPassManager):
    """
    Pass manager for the HTP backend.

    Extends QnnPassManager with HTP-specific graph transformations.
    """

    @classmethod
    def get_default_pass_activations(cls):
        # Reciprocal no longer appears at to_edge stage as it is decomposed in the export/annotation pipeline.
        # The current change is intended to proactively prepare for the upcoming deprecation of the export pipeline.
        pass_activations = super().get_default_pass_activations()
        pass_activations.extend([(DecomposeReciprocal, True)])
        pass_activations.append((TagPositionalSinCos, True))
        return pass_activations

    @classmethod
    def get_passes_dependency_for_capture_program(cls):
        # Reciprocal no longer appears at to_edge stage as it is decomposed in the export/annotation pipeline.
        # The current change is intended to proactively prepare for the upcoming deprecation of the export pipeline.
        deps = super().get_passes_dependency_for_capture_program()
        deps.update({DecomposeReciprocal: [RemoveRedundancy]})
        # Last before ResolveDebugHandle, so it tags the graph the partitioner sees.
        deps[TagPositionalSinCos] = [
            p
            for p, _ in cls.get_default_pass_activations()
            if p not in (TagPositionalSinCos, ResolveDebugHandle)
        ]
        deps[ResolveDebugHandle] = deps[ResolveDebugHandle] + [TagPositionalSinCos]
        return deps

    def _validate_edge_passes(self, passes: List[type]) -> None:
        super()._validate_edge_passes(passes)
        # TagPositionalSinCos moves the int32 casts I64toI32 adds together with
        # their int64 producers, so it must see them.
        order = {p: i for i, p in enumerate(passes)}
        if (
            TagPositionalSinCos in order
            and I64toI32 in order
            and order[TagPositionalSinCos] < order[I64toI32]
        ):
            raise ValueError(
                "TagPositionalSinCos must run after I64toI32. Start dep_table "
                f"from {type(self).__name__}.get_passes_dependency_for_capture_program() "
                "and edit it in place."
            )

    @classmethod
    def get_annotation_passes(cls, convert_linear_to_conv2d: bool = False):
        passes = [DecomposeReciprocal, RecomposeHadamard]
        passes.extend(
            super().get_annotation_passes(
                convert_linear_to_conv2d=convert_linear_to_conv2d,
            )
        )
        return passes

    @classmethod
    def get_export_passes(cls):
        # DecomposeReciprocal should be placed in the export pipeline, as it depends on
        # LiftConstantScalarOperands to lift the scalar operand.
        passes = [DecomposeReciprocal]
        passes.extend(super().get_export_passes())
        return passes
