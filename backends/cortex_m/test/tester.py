# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Any

import torch
from executorch.backends.arm.test.common import get_u55_compile_spec
from executorch.backends.arm.test.tester.arm_tester import Serialize
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager

from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import (
    Export,
    Quantize,
    RunPasses,
    StageType,
    ToEdge,
    ToExecutorch,
)

from executorch.exir import EdgeCompileConfig


class CortexMQuantize(Quantize):
    def __init__(self):
        quantizer = CortexMQuantizer()
        super().__init__(quantizer)


class CortexMToEdge(ToEdge):
    def __init__(self):
        config = EdgeCompileConfig(
            preserve_ops=[
                torch.ops.aten.linear.default,
                torch.ops.aten.hardsigmoid.default,
                torch.ops.aten.hardsigmoid_.default,
                torch.ops.aten.hardswish.default,
                torch.ops.aten.hardswish_.default,
            ]
        )
        super().__init__(config)


class CortexMRunPasses(RunPasses):
    def __init__(self):
        super().__init__(
            CortexMPassManager,
            CortexMPassManager.pass_list,
        )


class CortexMSerialize(Serialize):
    def __init__(self):
        compile_spec = get_u55_compile_spec()
        super().__init__(compile_spec, 1024)


cortex_m_stage_classes = {
    StageType.EXPORT: Export,
    StageType.QUANTIZE: CortexMQuantize,
    StageType.RUN_PASSES: CortexMRunPasses,
    StageType.SERIALIZE: Serialize,
    StageType.TO_EDGE: CortexMToEdge,
    StageType.TO_EXECUTORCH: ToExecutorch,
    StageType.SERIALIZE: CortexMSerialize,
}


class CortexMTester(TesterBase):
    def __init__(self, module, example_inputs):
        super().__init__(module, example_inputs, cortex_m_stage_classes)

    def test_dialect(self, ops_before_transforms, ops_after_transforms, qtol=0):
        """
        Test the python dialect op implementation.
        """
        self.quantize()
        self.export()
        self.to_edge()
        self.check_count(ops_before_transforms)
        self.run_passes()
        self.check_count(ops_after_transforms)
        self.run_method_and_compare_outputs(inputs=self.example_inputs, qtol=qtol)

    def test_implementation(self, qtol=0):
        """
        Test the optimized op implementation in simulation
        """
        self.quantize()
        self.export()
        self.to_edge()
        self.run_passes()
        self.to_executorch()
        self.serialize()
        self.run_method_and_compare_outputs(inputs=self.example_inputs, qtol=qtol)


@dataclass
class McuTestCase:
    model: torch.nn.Module
    example_inputs: tuple[Any]


def ramp_tensor(start: int, end: int, shape: tuple[int]) -> torch.Tensor:
    return torch.linspace(start, end, steps=torch.prod(torch.tensor(shape))).reshape(
        shape
    )
