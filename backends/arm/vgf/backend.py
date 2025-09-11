# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# Main implementation of AoT flow to partition and preprocess for VGF target
# backends. This flow converts via TOSA, to an encoding of TOSA known as VGF
# this form is used where the final JIT compile is performed on target (in the
# runtime delegate executorch::runtime::BackendInterface::init
#

import logging
import os
import subprocess
import tempfile
from typing import final, List

from executorch.backends.arm.tosa.backend import (
    arm_get_first_delegation_tag,
    TOSABackend,
)
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

# debug functionality
logger = logging.getLogger(__name__)


@final
class VgfBackend(BackendDetails):
    """
    BackendDetails subclass for delegation to VGF compatible devices. This enables
    encapsulated TOSA on target device and JIT compilation on suitable platforms.
    """

    @staticmethod
    def _compile_tosa_flatbuffer(
        tosa_flatbuffer: bytes,
        compile_spec: List[CompileSpec],
        tag_name: str = "",
    ) -> bytes:
        """
        Static helper method to do the compilation of the TOSA flatbuffer
        representation to a target specific binary stream.
        """
        compile_flags = []
        artifact_path = None
        for spec in compile_spec:
            if spec.key == "compile_flags":
                compile_flags.append(spec.value.decode())
            if spec.key == "debug_artifact_path":
                artifact_path = spec.value.decode()

        # Pass on the TOSA flatbuffer to the vgf compiler.
        binary = vgf_compile(tosa_flatbuffer, compile_flags, artifact_path, tag_name)
        return binary

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logger.info(f"{VgfBackend.__name__} preprocess")

        # deduce TOSA compile_spec from VGF compile spec. We get a new
        # compile spec list, containing only elements relevant for the
        # TOSABackend.
        tosa_compile_spec = TOSABackend.filter_tosa_compile_specs(compile_spec)

        # Backends doesn't allow inheritance, as stated in comments in exir/backend/backend_api.py
        # ('All backend implementation are final...'), so use composition instead.
        # preprocess returns the serialized TOSA flatbuffer in .processed_bytes,
        # which can be passed on to next compilation step.
        tosa_preprocess = TOSABackend.preprocess(edge_program, tosa_compile_spec)

        tag_name = arm_get_first_delegation_tag(edge_program.graph_module)

        binary = VgfBackend._compile_tosa_flatbuffer(
            tosa_preprocess.processed_bytes, compile_spec, tag_name
        )

        return PreprocessResult(processed_bytes=binary)


def vgf_compile(
    tosa_flatbuffer: bytes,
    compile_flags: List[str],
    artifact_path: str | None = None,
    tag_name: str = "",
):
    with tempfile.TemporaryDirectory() as tmpdir:

        # We currently write out a flatbuffer as input to the converter
        tosaname = f"output_{tag_name}.tosa"
        tosa_path = os.path.join(tmpdir, tosaname)
        with open(tosa_path, "wb") as f:
            f.write(tosa_flatbuffer)

        additional_flags = " ".join(compile_flags)
        vgf_path = tosa_path + ".vgf"
        conversion_command = (
            f"model-converter {additional_flags} -i {tosa_path} -o {vgf_path}"
        )
        try:
            subprocess.run(
                [conversion_command], shell=True, check=True, capture_output=True
            )
        except subprocess.CalledProcessError as process_error:
            raise RuntimeError(
                f"Vgf compiler ('{conversion_command}') failed with error:\n \
                {process_error.stderr.decode()}\n \
                Stdout:\n{process_error.stdout.decode()}"
            )

        if artifact_path is not None:
            logger.info(f"Emitting debug output to: {vgf_path=}")
            os.makedirs(artifact_path, exist_ok=True)
            cp = f"cp {vgf_path} {artifact_path}"
            subprocess.run(cp, shell=True, check=True, capture_output=False)

        vgf_bytes = open(vgf_path, "rb").read()
        return vgf_bytes
