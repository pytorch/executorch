# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.common.pipeline_config import (  # noqa: unused
    ArmPassPipelineConfig,
)
from executorch.backends.arm.tosa import TosaSpecification


class TosaCompileSpec(ArmCompileSpec):
    """Normalize and store the provided TOSA specification.

    Args:
        tosa_spec (TosaSpecification | str): Target spec object or version
            string supported by ``TosaSpecification.create_from_string``.

    """

    def __init__(self, tosa_spec: TosaSpecification | str):
        if isinstance(tosa_spec, str):
            tosa_spec = TosaSpecification.create_from_string(tosa_spec)
        self._set_compile_specs(tosa_spec, [])
        self.validate()

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

    @classmethod
    def from_list_hook(cls, compile_spec, specs: dict[str, str]):
        super().from_list_hook(compile_spec, specs)

    def _create_default_pipeline_config(self):
        config = super()._create_default_pipeline_config()
        return config
