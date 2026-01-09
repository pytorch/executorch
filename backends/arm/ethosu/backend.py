# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#
"""Ahead-of-time Arm Ethos-U backend built on the shared TOSA pipeline."""

import logging
from typing import final, List

from executorch.backends.arm.arm_vela import vela_compile
from executorch.backends.arm.ethosu.compile_spec import EthosUCompileSpec

from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

# debug functionality
logger = logging.getLogger(__name__)


@final
class EthosUBackend(BackendDetails):
    """BackendDetails subclass for delegation to Ethos-U.

    Deduce the TOSA lowering from the compile spec list by filtering out the
    compile spec values that are of interest for the TOSABackend.

    """

    @staticmethod
    def _compile_tosa_flatbuffer(
        tosa_flatbuffer: bytes, compile_spec: EthosUCompileSpec
    ) -> bytes:
        """Compile a TOSA flatbuffer into a target-specific binary stream.

        Args:
            tosa_flatbuffer (bytes): Serialized TOSA graph produced by
                ``TOSABackend``.
            compile_spec (EthosUCompileSpec): Compile specification providing
                Vela flags and intermediate paths.

        Returns:
            bytes: Target-specific binary stream produced by Vela.

        """
        compile_flags = compile_spec.compiler_flags

        if len(compile_flags) == 0:
            # Not testing for compile_flags correctness here, just that they are
            # present. The compiler will give errors if they are not valid.
            raise RuntimeError(
                "compile_flags are required in the CompileSpec list for EthosUBackend"
            )

        # Vela tooling only supports flatbuffers up to 2 GiB.
        max_flatbuffer_size = 2 * 1024 * 1024 * 1024
        flatbuffer_size = len(tosa_flatbuffer)
        if flatbuffer_size > max_flatbuffer_size:
            raise RuntimeError(
                "TOSA flatbuffer is too large for Vela "
                f"({flatbuffer_size} bytes > {max_flatbuffer_size} bytes limit)."
            )

        # Pass on the TOSA flatbuffer to the vela compiler.
        binary = vela_compile(
            tosa_flatbuffer,
            compile_flags,
            verbose=logger.getEffectiveLevel() <= logging.INFO,
            intermediate_path=compile_spec.get_intermediate_path(),
        )
        return binary

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Lower the exported program and compile it for an Ethos-U target.

        Args:
            edge_program (ExportedProgram): Program to lower to Ethos-U.
            compile_specs (List[CompileSpec]): Serialized Ethos-U compile specs
                supplied by the frontend.

        Returns:
            PreprocessResult: Result containing the compiled Ethos-U binary.

        """
        logger.info(f"{EthosUBackend.__name__} preprocess")

        compile_spec = EthosUCompileSpec.from_list(compile_specs)
        # deduce TOSA compile_spec from Ethos-U compile spec. We get a new
        # compile spec list, containing only elements relevant for the
        # TOSABackend.
        tosa_compile_spec = TOSABackend.filter_tosa_compile_specs(compile_spec)

        # Backends doesn't allow inheritance, as stated in comments in exir/backend/backend_api.py
        # ('All backend implementation are final...'), so use composition instead.
        # preprocess returns the serialized TOSA flatbuffer in .processed_bytes,
        # which can be passed on to next compilation step.
        tosa_preprocess = TOSABackend._preprocess(edge_program, tosa_compile_spec)

        binary = EthosUBackend._compile_tosa_flatbuffer(
            tosa_preprocess.processed_bytes, compile_spec
        )

        return PreprocessResult(processed_bytes=binary)
