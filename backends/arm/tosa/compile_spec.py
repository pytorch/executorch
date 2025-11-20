# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.tosa import TosaSpecification


class TosaCompileSpec(ArmCompileSpec):
    """Arm-specific compile spec capturing TOSA serializer requirements."""

    def __init__(self, tosa_spec: TosaSpecification | str):
        """Normalize and store the provided TOSA specification.

        Args:
            tosa_spec (TosaSpecification | str): Target spec object or version
                string supported by :meth:`TosaSpecification.create_from_string`.

        """
        if isinstance(tosa_spec, str):
            tosa_spec = TosaSpecification.create_from_string(tosa_spec)
        self._set_compile_specs(tosa_spec, [])

    def validate(self):
        """Ensure that no unsupported compiler flags were supplied."""
        if len(self.compiler_flags) != 0:
            raise ValueError(
                f"TosaCompileSpec can't have compiler flags, got {self.compiler_flags}"
            )
        pass

    @classmethod
    def get_output_format(cls) -> str:
        """Return the artifact format emitted by this compile spec."""
        return "tosa"
