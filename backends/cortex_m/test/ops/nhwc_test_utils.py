# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.passes.scratch_buffer_sizes import (
    required_cmsis_nn_buffer_sizes,
)
from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.test.harness.stages import RunPasses, StageType


def int8_values(shape):
    values = torch.arange(math.prod(shape), dtype=torch.int32)
    return (values.remainder(7) - 3).to(torch.int8).reshape(shape)


def run_on_fvp(module, x, target, target_config, scratch_count, atol=0):
    sizing_inputs = (x,) + tuple(
        torch.empty(0, dtype=torch.uint8) for _ in range(scratch_count)
    )
    tester = CortexMTester(module, sizing_inputs, target_config=target_config)
    tester.export().to_edge()
    program = tester.get_artifact(StageType.TO_EDGE).exported_program()
    [node] = [n for n in program.graph.nodes if n.target == target]
    scratch_sizes = required_cmsis_nn_buffer_sizes(node, target_config.backend) or []
    assert len(scratch_sizes) == scratch_count

    inputs = (x,) + tuple(
        torch.empty(size, dtype=torch.uint8) for size in scratch_sizes
    )
    tester = CortexMTester(module, inputs, target_config=target_config)
    tester.export().to_edge()
    tester.run_passes(RunPasses(CortexMPassManager, pass_list=[]))
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=inputs, atol=atol)
