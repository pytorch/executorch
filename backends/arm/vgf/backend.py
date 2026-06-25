# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#
# Main implementation of AoT flow to partition and preprocess for VGF target
# backends. This flow converts via TOSA, to an encoding of TOSA known as VGF
# this form is used where the final JIT compile is performed on target (in the
# runtime delegate executorch::runtime::BackendInterface::init
#
"""Ahead-of-time Arm VGF backend built on the shared TOSA pipeline."""

import logging
import os  # nosec B404 - used alongside subprocess for tool invocation
import shutil
import subprocess  # nosec B404 - required to drive external converter CLI
import tempfile
from dataclasses import dataclass
from typing import Any, final, List

from executorch.backends.arm._passes import RewriteConvPass
from executorch.backends.arm._passes.arm_pass_manager import (
    _registered_pass_insertions,
    PassInsertions,
    register_pass_insertions_before,
)
from executorch.backends.arm.tosa.backend import (  # type: ignore[import-not-found]
    arm_get_first_delegation_tag,
    TOSABackend,
)
from executorch.backends.arm.vgf._passes.rewrite_grid_sampler_to_tosa_custom import (  # type: ignore[import-not-found]
    RewriteGridSamplerToTosaCustomPass,
)

from executorch.backends.arm.vgf.compile_spec import (  # type: ignore[import-not-found]
    VgfCompileSpec,
)
from executorch.backends.arm.vgf.model_converter import (  # type: ignore[import-not-found]
    model_converter_env,
    require_model_converter_executable,
)
from executorch.exir.backend.backend_details import (  # type: ignore[import-not-found]
    BackendDetails,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import (  # type: ignore[import-not-found]
    CompileSpec,
)
from torch.export.exported_program import ExportedProgram

# debug functionality
logger = logging.getLogger(__name__)

STATUS_OK = "PASS"
STATUS_FAIL = "FAIL"
VGF_BACKEND_NAME = "VgfBackend"


@dataclass(frozen=True)
class VgfRuntimeEnvironmentCheck:
    """One VGF runtime backend environment preflight result.

    This lives next to the Python VGF backend name and backend implementation,
    while importing the actual ExecuTorch runtime lazily so AoT import behavior
    remains unchanged.

    """

    name: str
    status: str
    detail: str
    action: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != STATUS_FAIL

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "action": self.action,
        }


def _load_runtime() -> Any:
    from executorch.runtime import Runtime

    return Runtime.get()


def check_vgf_runtime_backend_environment() -> VgfRuntimeEnvironmentCheck:
    """Check whether the installed runtime exposes the VGF backend."""

    try:
        runtime = _load_runtime()
    except Exception as exc:
        return VgfRuntimeEnvironmentCheck(
            "VGF runtime backend",
            STATUS_FAIL,
            f"Could not initialize executorch.runtime.Runtime: {exc}",
            "Install or rebuild ExecuTorch with runtime pybindings. For source "
            "builds, enable the VGF runtime backend and reinstall the package.",
        )

    try:
        registered_backend_names = list(
            runtime.backend_registry.registered_backend_names
        )
        is_available = runtime.backend_registry.is_available(
            backend_name=VGF_BACKEND_NAME
        )
    except Exception as exc:
        return VgfRuntimeEnvironmentCheck(
            "VGF runtime backend",
            STATUS_FAIL,
            f"Runtime backend registry query failed: {exc}",
            "Reinstall or rebuild ExecuTorch with backend registry pybindings.",
        )

    if is_available:
        return VgfRuntimeEnvironmentCheck(
            "VGF runtime backend",
            STATUS_OK,
            f"{VGF_BACKEND_NAME} is available in the runtime backend registry.",
        )

    rendered = ", ".join(registered_backend_names[:20])
    if len(registered_backend_names) > 20:
        rendered += ", ..."

    return VgfRuntimeEnvironmentCheck(
        "VGF runtime backend",
        STATUS_FAIL,
        f"{VGF_BACKEND_NAME} is not available. Registered backends: "
        f"{rendered or '<none>'}.",
        "Use a runtime build/package that includes the VGF backend. For source "
        "builds, configure with -DEXECUTORCH_BUILD_VGF=ON and reinstall.",
    )


def _register_grid_sampler_rewrite_pass() -> None:
    """Register VGF-only custom shader lowering passes."""
    existing_insertions = _registered_pass_insertions.get(RewriteConvPass)
    if existing_insertions is not None and any(
        isinstance(pass_, RewriteGridSamplerToTosaCustomPass)
        for pass_ in existing_insertions.before_passes
    ):
        return
    register_pass_insertions_before(
        RewriteConvPass,
        [RewriteGridSamplerToTosaCustomPass()],
    )


def _snapshot_registered_pass_insertions() -> dict[type, PassInsertions]:
    return {
        pass_type: PassInsertions(
            before_passes=list(insertions.before_passes),
            after_passes=list(insertions.after_passes),
        )
        for pass_type, insertions in _registered_pass_insertions.items()
    }


def _restore_registered_pass_insertions(
    snapshot: dict[type, PassInsertions],
) -> None:
    _registered_pass_insertions.clear()
    _registered_pass_insertions.update(snapshot)


@final
class VgfBackend(BackendDetails):
    """BackendDetails subclass for delegation to VGF compatible devices.

    This enables encapsulated TOSA on target device and JIT compilation on
    suitable platforms.

    """

    @staticmethod
    def _compile_tosa_flatbuffer(
        tosa_flatbuffer: bytes,
        compile_spec: VgfCompileSpec,
        tag_name: str = "",
    ) -> bytes:
        """Compile a TOSA flatbuffer into a target-specific binary stream.

        Args:
            tosa_flatbuffer (bytes): Serialized TOSA graph produced by
                ``TOSABackend``.
            compile_spec (VgfCompileSpec): Compile specification providing
                converter flags and artifact paths.
            tag_name (str): Optional suffix used when producing debug outputs.

        Returns:
            bytes: Target-specific VGF binary stream.

        """
        compile_flags = compile_spec.compiler_flags
        artifact_path = compile_spec._get_intermediate_path()
        # Pass on the TOSA flatbuffer to the vgf compiler.
        binary = vgf_compile(tosa_flatbuffer, compile_flags, artifact_path, tag_name)
        return binary

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Lower the exported program and compile it for a VGF target.

        Args:
            edge_program (ExportedProgram): Program to lower to VGF.
            compile_specs (List[CompileSpec]): Serialized VGF compile specs
                supplied by the frontend.

        Returns:
            PreprocessResult: Result containing the compiled VGF binary.

        """
        logger.info(f"{VgfBackend.__name__} preprocess")

        insertions_snapshot = _snapshot_registered_pass_insertions()
        try:
            _register_grid_sampler_rewrite_pass()
            compile_spec = VgfCompileSpec._from_list(compile_specs)
            # deduce TOSA compile_spec from VGF compile spec. We get a new
            # compile spec list, containing only elements relevant for the
            # TOSABackend.
            tosa_compile_spec = TOSABackend.filter_tosa_compile_specs(compile_spec)

            # Backends doesn't allow inheritance, as stated in comments in exir/backend/backend_api.py
            # ('All backend implementation are final...'), so use composition instead.
            # preprocess returns the serialized TOSA flatbuffer in .processed_bytes,
            # which can be passed on to next compilation step.
            tosa_preprocess = TOSABackend._preprocess(edge_program, tosa_compile_spec)

            tag_name = arm_get_first_delegation_tag(edge_program.graph_module)

            binary = VgfBackend._compile_tosa_flatbuffer(
                tosa_preprocess.processed_bytes, compile_spec, tag_name
            )
        finally:
            _restore_registered_pass_insertions(insertions_snapshot)

        return PreprocessResult(processed_bytes=binary)


def vgf_compile(
    tosa_flatbuffer: bytes,
    compile_flags: List[str],
    artifact_path: str | None = None,
    tag_name: str = "",
):
    """Invoke the VGF compiler to convert a TOSA flatbuffer.

    Args:
        tosa_flatbuffer (bytes): Serialized TOSA graph produced by
            ``TOSABackend``.
        compile_flags (List[str]): Command-line flags forwarded to
            ``model-converter``.
        artifact_path (str | None): Directory where debug artifacts are saved.
        tag_name (str): Optional suffix used when producing debug outputs.

    Returns:
        bytes: Compiled VGF binary emitted by ``model-converter``.

    """
    with tempfile.TemporaryDirectory() as tmpdir:

        # We currently write out a flatbuffer as input to the converter
        tosaname = f"output_{tag_name}.tosa"
        tosa_path = os.path.join(tmpdir, tosaname)
        with open(tosa_path, "wb") as f:
            f.write(tosa_flatbuffer)

        compile_flags = [f for f in compile_flags if f and f.strip()]
        converter_binary = str(require_model_converter_executable())
        vgf_path = tosa_path + ".vgf"
        conversion_command = [
            converter_binary,
            *compile_flags,
            "-i",
            tosa_path,
            "-o",
            vgf_path,
        ]
        try:
            subprocess.run(  # nosec B602, B603 - shell invocation constrained to trusted converter binary with trusted inputs
                conversion_command,
                shell=False,
                check=True,
                capture_output=True,
                env=model_converter_env(),
            )
        except subprocess.CalledProcessError as process_error:
            conversion_command_str = " ".join(conversion_command)
            raise RuntimeError(
                f"Vgf compiler ('{conversion_command_str}') failed with error:\n \
                {process_error.stderr.decode()}\n \
                Stdout:\n{process_error.stdout.decode()}"
            )

        if artifact_path:
            logger.info(f"Emitting debug output to: {vgf_path=}")
            os.makedirs(artifact_path, exist_ok=True)
            shutil.copy2(vgf_path, artifact_path)

        vgf_bytes = open(vgf_path, "rb").read()
        return vgf_bytes
