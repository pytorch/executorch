# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.util._factory import create_partitioner
from executorch.exir import EdgeCompileConfig
from executorch.exir.pass_manager import PassType
from executorch.export import (  # type: ignore[import-untyped]
    ExportRecipe,
    LoweringRecipe,
    QuantizationRecipe,
)


@dataclass
class TargetRecipe:
    """Contains target-level export configuration."""

    compile_spec: ArmCompileSpec
    edge_compile_config: EdgeCompileConfig = field(
        default_factory=lambda: EdgeCompileConfig(_check_ir_validity=False)
    )
    edge_transform_passes: List[PassType] = field(default_factory=lambda: [])


class ArmExportRecipe(ExportRecipe):
    """Wraps ExportRecipe to provide the constructor we want and easy access to some variables."""

    def __init__(
        self,
        name,
        target_recipe: TargetRecipe,
        quantization_recipe: QuantizationRecipe | None,
    ):
        self.compile_spec = target_recipe.compile_spec
        self.edge_transform_passes = target_recipe.edge_transform_passes

        lowering_recipe = LoweringRecipe(
            [create_partitioner(self.compile_spec)],
            edge_transform_passes=[lambda _, __: target_recipe.edge_transform_passes],
            edge_compile_config=target_recipe.edge_compile_config,
        )
        super().__init__(
            name=name,
            quantization_recipe=quantization_recipe,
            lowering_recipe=lowering_recipe,
            executorch_backend_config=None,
            pipeline_stages=None,
        )
