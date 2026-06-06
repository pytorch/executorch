# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import TYPE_CHECKING

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.tosa import (  # type: ignore[import-not-found]
    TosaSpecification,
)

if TYPE_CHECKING:
    from executorch.backends.arm.vgf.check_env import VgfEnvironmentReport

# debug functionality
logger = logging.getLogger(__name__)


class VgfCompileSpec(ArmCompileSpec):
    """Normalise inputs and populate the underlying Arm compile spec.

    Args:
        tosa_spec (TosaSpecification | str | None): TOSA specification to
            target. Strings are parsed via ``TosaSpecification.create_from_string``.
            Defaults to ``"TOSA-1.0+FP+INT+int4+int16"``.
        compiler_flags (list[str] | None): Optional converter-backend flags.

    """

    def __init__(
        self,
        tosa_spec: TosaSpecification | str | None = None,
        compiler_flags: list[str] | None = None,
    ):
        if tosa_spec is None:
            tosa_spec = TosaSpecification.create_from_string(
                "TOSA-1.0+FP+INT+int4+int16"
            )
        elif isinstance(tosa_spec, str):
            tosa_spec = TosaSpecification.create_from_string(tosa_spec)

        if compiler_flags is None:
            compiler_flags = []
        self._set_compile_specs(tosa_spec, compiler_flags)
        self._validate()

    def _validate(self):
        """Validate the configuration against VGF-supported TOSA profiles."""
        tosa_version = self.tosa_spec.version  # type: ignore[attr-defined]
        tosa_profiles = self.tosa_spec.profiles  # type: ignore[attr-defined]

        if tosa_version.major != 1:
            raise ValueError(
                "Arm backend only supports converter-backend for TOSA version 1. "
                f"Invalid TOSA version: {tosa_version}"
            )

        if "FP" not in tosa_profiles and "INT" not in tosa_profiles:
            raise ValueError(
                "Arm backend only supports converter-backend for FP and/or INT. "
                f"Invalid TOSA profile: {tosa_profiles}"
            )

    def validate_environment(
        self,
        build_dir: str | None = None,
        *,
        require_runtime_build: bool = False,
    ) -> "VgfEnvironmentReport":
        """Run VGF environment preflight checks.

        By default this validates only AoT/export prerequisites. Runtime and
        source-build diagnostics are intentionally explicit in check_env.py.

        Args:
            build_dir: Optional source-build CMake build directory or
                CMakeCache.txt path.
            require_runtime_build: If true, run source-build diagnostics instead
                of the default AoT check.

        Returns:
            VgfEnvironmentReport: Structured check report.

        Raises:
            RuntimeError: If any required check fails.

        """
        from executorch.backends.arm.vgf.check_env import (
            check_vgf_aot_environment,
            check_vgf_source_build_environment,
        )

        if build_dir is not None or require_runtime_build:
            report = check_vgf_source_build_environment(build_dir=build_dir)
        else:
            report = check_vgf_aot_environment()

        report.raise_for_errors()
        return report

    @classmethod
    def _get_output_format(cls) -> str:
        """Return the artifact format emitted by this compile spec."""
        return "vgf"
