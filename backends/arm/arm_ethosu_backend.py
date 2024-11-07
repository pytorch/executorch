# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#

import logging
from typing import final, List

from executorch.backends.arm.arm_tosa_backend import ArmTOSABackend

from executorch.backends.arm.arm_vela import vela_compile
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

# debug functionality
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@final
class ArmEthosUBackend(BackendDetails):
    """
    BackendDetails subclass for delegation to Ethos-U. Deduce the TOSA lowering from
    the compile spec list by filtering out the compile spec values that are of interest
    for the ArmTOSABackend.
    """

    @staticmethod
    def preprocess(  # noqa: C901
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logger.info(f"{ArmEthosUBackend.__name__} preprocess")

        # deduce TOSA compile_spec from Ethos-U compile spec. We get a new
        # compile spec list, containing only elements relevant for the
        # ArmTOSABackend.
        tosa_compile_spec = ArmTOSABackend.filter_tosa_compile_specs(compile_spec)

        # Backends doesn't allow inheritance, as stated in comments in exir/backend/backend_api.py
        # ('All backend implementation are final...'), so use composition instead.
        # preprocess returns the serialized TOSA flatbuffer in .processed_bytes,
        # which can be passed on to next compilation step.
        tosa_preprocess = ArmTOSABackend.preprocess(edge_program, tosa_compile_spec)

        compile_flags = []
        for spec in compile_spec:
            if spec.key == "compile_flags":
                compile_flags.append(spec.value.decode())

        # Pass on the TOSA flatbuffer to the vela compiler.
        binary = vela_compile(tosa_preprocess.processed_bytes, compile_flags)

        return PreprocessResult(processed_bytes=binary)
