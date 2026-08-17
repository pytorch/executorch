# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cortex-M flow for the backend test suite.

Every case is serialized and executed on the Corstone-300 FVP with the
semihosting runner from backends/cortex_m/test/build_test_runner.sh. That runner
registers the cortex_m kernels along with the portable fallbacks listed in its
ops_list, so an operator the backend does not lower still runs when the list
happens to carry one.

The kernels are CMSIS-NN compiled for the device and cannot run on a host CPU,
so the simulator is the only place to execute them. The Arm Ethos-U flows reach
the Corstone FVP the same way. flow.py leaves the flow unregistered when the FVP
is not on PATH.

Comparison uses the suite's fixed atol of 1e-1, which is not tuned for int8, so
a mismatch is not by itself evidence of a lowering bug.
"""

import torch

from executorch.backends.cortex_m.test.tester import CortexMQuantize, CortexMTester
from executorch.backends.test.suite.flow import TestFlow


def _create_cortex_m_tester(model, inputs, **kwargs) -> CortexMTester:
    # The CMSIS-NN kernels are NHWC, and CortexMConv2DCheck rejects any pattern
    # whose tensors are not channels_last. The suite hands over contiguous
    # tensors, which silently costs every convolution in the model.
    inputs = tuple(
        (
            t.to(memory_format=torch.channels_last)
            if isinstance(t, torch.Tensor) and t.dim() == 4
            else t
        )
        for t in inputs
    )
    # The Serialize stage is also the stage that invokes the ELF, so its timeout
    # is the FVP's --timelimit rather than a serialization budget, and 120s is
    # not enough for an ImageNet-sized model on an M55 with no NPU. Kept under
    # the suite conftest's 1200s pytest timeout so the FVP reports the overrun
    # instead of pytest killing the process.
    return CortexMTester(model, inputs, timeout=900, **kwargs)


# Models that cannot run on the FVP runner at all, so there is nothing for the
# report to record. A skipped test produces no row, which is why the sizes are
# written down here.
CORTEX_M_SKIPS = [
    # Over the runner's 60 MiB pool even fully quantized: 86.6M and 68.9M
    # parameters. Raising ET_ARM_BAREMETAL_SEMIHOSTING_FILE_ALLOCATOR_POOL_SIZE
    # is the only thing that would change that.
    "test_vit_b_16",
    "test_wide_resnet50_2",
    # The program is under a megabyte; it is the activation arena that does not
    # fit, and it crosses the pool at a sequence length of about 353. The test
    # draws lengths from randint(1, 400), so it fits roughly one run in three.
    "test_conformer",
]

# Models that run to completion and fail. They stay in the suite so the report
# keeps recording how they fail; the marker only keeps a known state from
# turning the job red, and it is strict, so anything fixed reports XPASS and
# this list has to shrink.
CORTEX_M_XFAILS = [
    # Whatever a model leaves unlowered runs on the portable kernels compiled
    # into the runner, and ops_list in build_test_runner.sh is a fixed list.
    "test_convnext_small",
    "test_densenet161",
    "test_maxvit_t",
    "test_mnasnet1_0",
    "test_shufflenet_v2_x1_0",
    # 4.00 bytes per parameter: none of the convolutions lower, so the weights
    # stay fp32 and the 100 MiB program overruns the pool. Lowering conv1d
    # brings it to 26 MiB, which fits.
    "test_wav2letter",
    # Runs on target and does not match.
    "test_inception_v3",
    "test_resnet50",
    "test_squeezenet1_1",
    # AtenToCortexMPass rejects the graph.
    "test_swin_v2_t",
]


def _create_cortex_m_flow() -> TestFlow:
    return TestFlow(
        "cortex_m",
        backend="cortex_m",
        tester_factory=_create_cortex_m_tester,
        quantize=True,
        quantize_stage_factory=CortexMQuantize,
        is_delegated=False,
        param_skip_reasons={
            "use_dynamic_shapes": {
                True: "Cortex-M lowers for a fixed shape; the CMSIS-NN kernels "
                "take their dimensions from the graph."
            }
        },
        skip_patterns=CORTEX_M_SKIPS,
        xfail_patterns=CORTEX_M_XFAILS,
    )


CORTEX_M_TEST_FLOW = _create_cortex_m_flow()
