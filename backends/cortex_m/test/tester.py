# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import torch
from executorch.backends.arm.test.common import get_u55_compile_spec
from executorch.backends.arm.test.tester.arm_tester import Serialize
from executorch.backends.cortex_m.edge_compile_config import (
    cortex_m_edge_compile_config,
)
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import (
    Export,
    Quantize,
    RunPasses,
    StageType,
    ToEdge,
    ToEdgeTransformAndLower,
    ToExecutorch,
)


class CortexMQuantize(Quantize):
    def __init__(self, calibration_samples=None):
        quantizer = CortexMQuantizer()
        super().__init__(quantizer, calibration_samples=calibration_samples)


class CortexMToEdge(ToEdge):
    def __init__(self):
        super().__init__(cortex_m_edge_compile_config())


class CortexMRunPasses(RunPasses):
    def __init__(self, target_config: Optional[CortexMTargetConfig] = None):
        target_config = target_config or CortexMTargetConfig(cpu=CortexM.M55)
        # The base RunPasses constructs the pass manager as `cls(ep, pass_list)`.
        # Pre-bind the target_config so it flows through that 2-arg call.
        super().__init__(
            partial(CortexMPassManager, target_config=target_config),  # type: ignore[arg-type]
            CortexMPassManager.pass_list,  # type: ignore[arg-type]
        )


class CortexMToEdgeTransformAndLower(ToEdgeTransformAndLower):
    """to_edge with no partitioner, then CortexMPassManager.

    Cortex-M rewrites edge operators in place rather than delegating a subgraph,
    so this is its equivalent of to_edge_transform_and_lower, which is the only
    lowering entry point the shared backend test suite drives.
    """

    def __init__(self, target_config: Optional[CortexMTargetConfig] = None):
        super().__init__(edge_compile_config=cortex_m_edge_compile_config())
        self._run_passes = CortexMRunPasses(target_config)

    def run(self, artifact, inputs=None, generate_etrecord: bool = False) -> None:
        super().run(artifact, inputs, generate_etrecord=generate_etrecord)
        self._run_passes.run(self.edge_dialect_program, inputs)  # type: ignore[arg-type]


class CortexMSerialize(Serialize):
    def __init__(
        self,
        target_config: Optional[CortexMTargetConfig] = None,
        timeout: int = 120,
    ):
        target_config = target_config or CortexMTargetConfig(cpu=CortexM.M55)
        compile_spec = get_u55_compile_spec()
        # Select the runner built for this target (build_test_runner.sh writes
        # one runner per target into a target-suffixed directory).
        super().__init__(
            compile_spec,
            None,
            timeout=timeout,
            build_dir_suffix=f"_{target_config.target_string}",
        )


cortex_m_stage_classes = {
    StageType.EXPORT: Export,
    StageType.QUANTIZE: CortexMQuantize,
    StageType.RUN_PASSES: CortexMRunPasses,
    StageType.TO_EDGE: CortexMToEdge,
    StageType.TO_EDGE_TRANSFORM_AND_LOWER: CortexMToEdgeTransformAndLower,
    StageType.TO_EXECUTORCH: ToExecutorch,
    StageType.SERIALIZE: CortexMSerialize,
}


class CortexMTester(TesterBase):
    def __init__(
        self,
        module,
        example_inputs,
        target_config: Optional[CortexMTargetConfig] = None,
        timeout: int = 120,
    ):
        if callable(example_inputs):
            resolved_example_inputs = example_inputs()
        else:
            resolved_example_inputs = example_inputs
        target_config = target_config or CortexMTargetConfig(cpu=CortexM.M55)
        stage_classes: dict[StageType, Callable[..., Any]] = dict(
            cortex_m_stage_classes
        )
        stage_classes[StageType.RUN_PASSES] = lambda: CortexMRunPasses(
            target_config=target_config
        )
        stage_classes[StageType.TO_EDGE_TRANSFORM_AND_LOWER] = (
            lambda: CortexMToEdgeTransformAndLower(target_config=target_config)
        )
        stage_classes[StageType.SERIALIZE] = lambda: CortexMSerialize(
            target_config=target_config, timeout=timeout
        )
        super().__init__(module, resolved_example_inputs, stage_classes)

    def test_dialect(
        self,
        ops_before_transforms,
        ops_after_transforms,
        qtol=0,
        atol=1e-03,
        calibration_samples=None,
    ):
        """
        Test the python dialect op implementation.
        """
        if calibration_samples is not None:
            quantization_stage = CortexMQuantize(
                calibration_samples=calibration_samples
            )
        else:
            quantization_stage = None

        self.quantize(quantization_stage)
        self.export()
        self.to_edge()
        self.check_count(ops_before_transforms)
        self.run_passes()
        self.check_count(ops_after_transforms)
        self.run_method_and_compare_outputs(
            inputs=self.example_inputs, qtol=qtol, atol=atol
        )

    def test_implementation(self, qtol=0, atol=1e-03, calibration_samples=None):
        """
        Test the optimized op implementation in simulation
        """

        if calibration_samples is not None:
            quantization_stage = CortexMQuantize(
                calibration_samples=calibration_samples
            )
        else:
            quantization_stage = None

        self.quantize(quantization_stage)
        self.export()
        self.to_edge()
        self.run_passes()
        self.to_executorch()
        self.serialize()
        self.run_method_and_compare_outputs(
            inputs=self.example_inputs, qtol=qtol, atol=atol
        )


@dataclass
class McuTestCase:
    model: torch.nn.Module
    example_inputs: tuple[Any, ...] | Callable[[], tuple[Any, ...]]

    def get_example_inputs(self) -> tuple[Any, ...]:
        if callable(self.example_inputs):
            return self.example_inputs()
        return self.example_inputs


def ramp_tensor(start: float, end: float, shape: tuple[int, ...]) -> torch.Tensor:
    steps = int(torch.prod(torch.tensor(shape)).item())
    return torch.linspace(start, end, steps=steps).reshape(shape)
