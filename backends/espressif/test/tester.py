# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.quantizer import EspNNQuantizer
from executorch.backends.test.harness import Tester
from executorch.backends.test.harness.stages import (
    Quantize,
    RunPasses,
    StageType,
    ToEdgeTransformAndLower,
)
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)


class EspRunPasses(RunPasses):
    def __init__(self):
        super().__init__(EspPassManager)

    def run(self, artifact: EdgeProgramManager, inputs=None) -> None:
        self.edge_or_aten_program = artifact.transform(EspPassManager())


class EspToEdgeTransformAndLower(ToEdgeTransformAndLower):
    def __init__(self):
        super().__init__(edge_compile_config=EdgeCompileConfig())

    def run(self, artifact, inputs=None, generate_etrecord: bool = False) -> None:
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact,
            compile_config=self.edge_compile_conf,
            transform_passes=EspPassManager(),
            generate_etrecord=generate_etrecord,
        )


class EspTester(Tester):
    def __init__(self, module, example_inputs, calibration_samples=None):
        stages = self.default_stage_classes()
        stages[StageType.QUANTIZE] = lambda: Quantize(
            EspNNQuantizer(), calibration_samples=calibration_samples
        )
        stages[StageType.RUN_PASSES] = EspRunPasses
        stages[StageType.TO_EDGE_TRANSFORM_AND_LOWER] = EspToEdgeTransformAndLower
        super().__init__(module, example_inputs, stages)
