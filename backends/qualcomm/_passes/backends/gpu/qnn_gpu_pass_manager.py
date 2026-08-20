# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm._passes import DecomposeReciprocal, RemoveRedundancy
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager


class QnnGpuPassManager(QnnPassManager):
    """
    Pass manager for the GPU backend.

    Extends QnnPassManager with GPU-specific graph transformations.
    """

    @classmethod
    def get_default_pass_activations(cls):
        # Reciprocal no longer appears at to_edge stage as it is decomposed in the export pipeline.
        # The current change is intended to proactively prepare for the upcoming deprecation of the export pipeline.
        pass_activations = super().get_default_pass_activations()
        pass_activations.extend([(DecomposeReciprocal, True)])
        return pass_activations

    @classmethod
    def get_passes_dependency_for_capture_program(cls):
        # Reciprocal no longer appears at to_edge stage as it is decomposed in the export pipeline.
        # The current change is intended to proactively prepare for the upcoming deprecation of the export pipeline.
        deps = super().get_passes_dependency_for_capture_program()
        deps.update({DecomposeReciprocal: [RemoveRedundancy]})
        return deps

    @classmethod
    def get_annotation_passes(cls):
        # The annotation pipeline is skipped for the GPU backend, as it does not
        # support quantized data types. Return an empty list to indicate a no-op.
        return []

    @classmethod
    def get_export_passes(
        cls,
        convert_linear_to_conv2d: bool = False,
    ):
        # DecomposeReciprocal should be placed in the export pipeline, as it depends on
        # LiftConstantScalarOperands to lift the scalar operand.
        passes = [DecomposeReciprocal]
        passes.extend(super().get_export_passes(convert_linear_to_conv2d))
        return passes
