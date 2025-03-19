# Copyright 2025 Arm Limited and/or its affiliates.
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

from executorch.backends.arm.arm_vela import vela_compile

from executorch.backends.arm.tosa_backend import TOSABackend
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

# debug functionality
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@final
class EthosUBackend(BackendDetails):
    """
    BackendDetails subclass for delegation to Ethos-U. Deduce the TOSA lowering from
    the compile spec list by filtering out the compile spec values that are of interest
    for the TOSABackend.
    """

    @staticmethod
    def _compile_tosa_flatbuffer(
        tosa_flatbuffer: bytes, compile_spec: list[CompileSpec]
    ) -> bytes:
        """
        Static helper method to do the compilation of the TOSA flatbuffer
        representation to a target specific binary stream.
        """
        compile_flags = []
        input_order = []
        for spec in compile_spec:
            if spec.key == "compile_flags":
                compile_flags.append(spec.value.decode())
            if spec.key == "input_order":
                input_order = list(map(int, spec.value.decode().split(",")))

        if len(compile_flags) == 0:
            # Not testing for compile_flags correctness here, just that they are
            # present. The compiler will give errors if they are not valid.
            raise RuntimeError(
                "compile_flags are required in the CompileSpec list for EthosUBackend"
            )

        # Pass on the TOSA flatbuffer to the vela compiler.
        binary = vela_compile(
            tosa_flatbuffer,
            compile_flags,
            input_order,
            verbose=logger.getEffectiveLevel() == logging.INFO,
        )
        return binary

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logger.info(f"{EthosUBackend.__name__} preprocess")

        # deduce TOSA compile_spec from Ethos-U compile spec. We get a new
        # compile spec list, containing only elements relevant for the
        # TOSABackend.
        tosa_compile_spec = TOSABackend.filter_tosa_compile_specs(compile_spec)

        # Backends doesn't allow inheritance, as stated in comments in exir/backend/backend_api.py
        # ('All backend implementation are final...'), so use composition instead.
        # preprocess returns the serialized TOSA flatbuffer in .processed_bytes,
        # which can be passed on to next compilation step.
        tosa_preprocess = TOSABackend.preprocess(edge_program, tosa_compile_spec)

        binary = EthosUBackend._compile_tosa_flatbuffer(
            tosa_preprocess.processed_bytes, compile_spec
        )

        return PreprocessResult(processed_bytes=binary)
