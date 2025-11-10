# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.tosa import (  # type: ignore[import-not-found]
    TosaSpecification,
)

# debug functionality
logger = logging.getLogger(__name__)


class VgfCompileSpec(ArmCompileSpec):
    """
    Compile spec for VGF compatible targets.

    Args:
        tosa_spec: TOSA specification that should be targeted.
        compiler_flags: Extra compiler flags for converter_backend.
    """

    def __init__(
        self,
        tosa_spec: TosaSpecification | str | None = None,
        compiler_flags: list[str] | None = None,
    ):
        if tosa_spec is None:
            tosa_spec = "TOSA-1.0+FP"
        if isinstance(tosa_spec, str):
            tosa_spec = TosaSpecification.create_from_string(tosa_spec)

        if compiler_flags is None:
            compiler_flags = []
        self._set_compile_specs(tosa_spec, compiler_flags)
        self.validate()

    def validate(self):
        """Throws an error if the compile spec is not valid."""
        tosa_version = self.tosa_spec.version  # type: ignore[attr-defined]
        tosa_profiles = self.tosa_spec.profiles  # type: ignore[attr-defined]

        if tosa_version.major != 1:
            raise ValueError(
                "Arm backend only supports converter-backend for TOSA version 1. "
                f"Invalid TOSA version: {tosa_version}"
            )

        if "FP" not in tosa_profiles and "INT" not in tosa_profiles:
            raise ValueError(
                "Arm backend only supports converter-backend for FP or INT. "
                f"Invalid TOSA profile: {tosa_profiles}"
            )

        if len(tosa_profiles) != 1:
            raise ValueError(
                "For now Arm backend only supports converter-backend for either FP or INT. "
                f"Invalid TOSA profile: {tosa_profiles}"
            )

    @classmethod
    def get_output_format(cls) -> str:
        return "vgf"
