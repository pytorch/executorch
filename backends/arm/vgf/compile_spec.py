# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.common.pipeline_config import (  # noqa: unused
    ArmPassPipelineConfig,
)
from executorch.backends.arm.tosa import (  # type: ignore[import-not-found]
    TosaSpecification,
)

# debug functionality
logger = logging.getLogger(__name__)


class VgfCompileSpec(ArmCompileSpec):
    """Normalise inputs and populate the underlying Arm compile spec.

    Args:
        tosa_spec (TosaSpecification | str | None): TOSA specification to
            target. Strings are parsed via ``TosaSpecification.create_from_string``.
            Defaults to ``"TOSA-1.0+FP+INT"``.
        compiler_flags (list[str] | None): Optional converter-backend flags.

    """

    def __init__(
        self,
        tosa_spec: TosaSpecification | str | None = None,
        compiler_flags: list[str] | None = None,
    ):
        if tosa_spec is None:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP+INT")
        elif isinstance(tosa_spec, str):
            tosa_spec = TosaSpecification.create_from_string(tosa_spec)

        if compiler_flags is None:
            compiler_flags = []
        self._set_compile_specs(tosa_spec, compiler_flags)
        self.validate()

    def validate(self):
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

    @classmethod
    def get_output_format(cls) -> str:
        """Return the artifact format emitted by this compile spec."""
        return "vgf"

    def _create_default_pipeline_config(self) -> ArmPassPipelineConfig:
        config = super()._create_default_pipeline_config()
        # GRPHCOMP-3140 / MLETORCH-1529
        config.disable_fuse_duplicate_users()
        return config
