# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, Optional

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
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from torch.export import ExportedProgram


class CortexMQuantize(Quantize):
    def __init__(self, calibration_samples=None, use_explicit_layout: bool = False):
        quantizer = CortexMQuantizer(use_explicit_layout=use_explicit_layout)
        super().__init__(quantizer, calibration_samples=calibration_samples)


class CortexMToEdge(ToEdge):
    def __init__(self):
        super().__init__(cortex_m_edge_compile_config())


class CortexMRunPasses(RunPasses):
    def __init__(
        self,
        target_config: Optional[CortexMTargetConfig] = None,
        use_explicit_layout: bool = False,
    ):
        super().__init__(CortexMPassManager)
        self.pass_manager = CortexMPassManager(
            target_config=target_config,
            use_explicit_layout=use_explicit_layout,
        )

    def run(self, artifact: EdgeProgramManager | ExportedProgram, inputs=None) -> None:
        if isinstance(artifact, EdgeProgramManager):
            self.edge_or_aten_program = artifact.transform(self.pass_manager)
        else:
            self.edge_or_aten_program = self.pass_manager(artifact).exported_program


class CortexMToEdgeTransformAndLower(ToEdgeTransformAndLower):
    def __init__(self, target_config: Optional[CortexMTargetConfig] = None):
        super().__init__(edge_compile_config=cortex_m_edge_compile_config())
        self.pass_manager = CortexMPassManager(target_config=target_config)

    def run(self, artifact, inputs=None, generate_etrecord: bool = False) -> None:
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact,
            compile_config=self.edge_compile_conf,
            transform_passes=self.pass_manager,
            generate_etrecord=generate_etrecord,
        )


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
        self.target_config = target_config
        stage_classes: dict[StageType, Callable[..., Any]] = dict(
            cortex_m_stage_classes
        )
        stage_classes[StageType.RUN_PASSES] = lambda use_explicit_layout=False: (
            CortexMRunPasses(
                target_config=target_config, use_explicit_layout=use_explicit_layout
            )
        )
        stage_classes[StageType.TO_EDGE_TRANSFORM_AND_LOWER] = lambda: (
            CortexMToEdgeTransformAndLower(target_config=target_config)
        )
        stage_classes[StageType.SERIALIZE] = lambda: CortexMSerialize(
            target_config=target_config, timeout=timeout
        )
        super().__init__(module, resolved_example_inputs, stage_classes)  # pyrefly: ignore [bad-argument-type]

    def test_dialect(
        self,
        ops_before_transforms,
        ops_after_transforms,
        qtol=0,
        atol=1e-03,
        calibration_samples=None,
        ops_absent_after_transforms=None,
        use_explicit_layout: bool = False,
        compare_outputs: bool = True,
    ):
        """
        Test the python dialect op implementation.
        """
        if calibration_samples is None and not use_explicit_layout:
            self.quantize()
        else:
            quantization_stage = cast(
                Quantize,
                self._get_default_stage(
                    StageType.QUANTIZE,
                    calibration_samples=calibration_samples,
                    use_explicit_layout=use_explicit_layout,
                ),
            )
            self.quantize(quantization_stage)
        self.export()
        self.to_edge()
        self.check_count(ops_before_transforms)
        if use_explicit_layout:
            self.run_passes(
                cast(
                    RunPasses,
                    self._get_default_stage(
                        StageType.RUN_PASSES, use_explicit_layout=True
                    ),
                )
            )
        else:
            self.run_passes()
        self.check_count(ops_after_transforms)
        if ops_absent_after_transforms:
            self.check_not(ops_absent_after_transforms)
        if compare_outputs:
            self.run_method_and_compare_outputs(
                inputs=self.example_inputs, qtol=qtol, atol=atol
            )

    def test_implementation(
        self,
        qtol=0,
        atol=1e-03,
        calibration_samples=None,
        use_explicit_layout: bool = False,
        compare_outputs: bool = True,
    ):
        """
        Test the optimized op implementation in simulation
        """

        if calibration_samples is None and not use_explicit_layout:
            self.quantize()
        else:
            self.quantize(
                cast(
                    Quantize,
                    self._get_default_stage(
                        StageType.QUANTIZE,
                        calibration_samples=calibration_samples,
                        use_explicit_layout=use_explicit_layout,
                    ),
                )
            )
        self.export()
        self.to_edge()
        if use_explicit_layout:
            self.run_passes(
                cast(
                    RunPasses,
                    self._get_default_stage(
                        StageType.RUN_PASSES, use_explicit_layout=True
                    ),
                )
            )
        else:
            self.run_passes()
        self.to_executorch()
        self.serialize()
        if compare_outputs:
            self.run_method_and_compare_outputs(
                inputs=self.example_inputs, qtol=qtol, atol=atol
            )


@dataclass
class McuTestCase:
    model: torch.nn.Module
    example_inputs: tuple[Any, ...] | Callable[[], tuple[Any, ...]]

    def get_example_inputs(self, use_explicit_layout: bool = False) -> tuple[Any, ...]:
        inputs = (
            self.example_inputs()
            if callable(self.example_inputs)
            else self.example_inputs
        )
        if use_explicit_layout:
            return tuple(
                (
                    input_value.clone(memory_format=torch.contiguous_format)
                    if isinstance(input_value, torch.Tensor)
                    else input_value
                )
                for input_value in inputs
            )
        return inputs


def ramp_tensor(start: float, end: float, shape: tuple[int, ...]) -> torch.Tensor:
    steps = int(torch.prod(torch.tensor(shape)).item())
    return torch.linspace(start, end, steps=steps).reshape(shape)
