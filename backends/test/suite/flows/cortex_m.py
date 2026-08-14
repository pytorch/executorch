# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cortex-M flow for the backend test suite.

Runs on the Corstone-300 FVP, using the runner from
backends/cortex_m/test/build_test_runner.sh. There is no host path:
EXECUTORCH_BUILD_CORTEX_M is off in the default preset, so the host runtime
carries no cortex_m kernels.

Read a pass here as "the model lowered and ran", not as an accuracy result.
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
    # An FVP run of an ImageNet-sized model exceeds Serialize's 120s default.
    return CortexMTester(model, inputs, timeout=1200, **kwargs)


# Models that do not run today. The reason each one fails is recorded here
# because the suite only reports that a test was skipped.
CORTEX_M_SKIPS = [
    # Whatever a model leaves unlowered runs on the portable kernels compiled
    # into the runner, and ops_list in build_test_runner.sh is a fixed list.
    "test_conformer",  # aten::native_layer_norm.out
    "test_convnext_small",  # aten::native_layer_norm.out
    "test_efficientnet_b4",  # aten::silu.out
    "test_efficientnet_v2_s",  # aten::silu.out
    "test_mnasnet1_0",  # dim_order_ops::_to_dim_order_copy.out
    "test_shufflenet_v2_x1_0",  # aten::split_with_sizes_copy.out
    # These two want a kernel the runner should not have to carry: the batch
    # norm should have been folded, and cortex_m::transpose already exists.
    "test_densenet161",  # aten::_native_batch_norm_legit_no_training.out
    "test_maxvit_t",  # aten::permute_copy.out
    # The .pte does not fit. The runner reads the whole program into a 60 MiB
    # pool before memory planning is consulted. wav2letter is 100 MiB because
    # its conv1d layers never lower, so the weights stay fp32.
    "test_vit_b_16",  # 84 MiB
    "test_wav2letter",  # 100 MiB
    "test_wide_resnet50_2",  # 109 MiB
    # Run on target and return corrupt values. Both are concatenation-heavy and
    # the portable NHWC cat kernel is known to corrupt results.
    "test_inception_v3",  # max error 7e11
    "test_squeezenet1_1",
    # Ordinary int8 error on a 1000-way logit vector, at 35 dB SNR, rather than
    # a wrong result -- but the suite's atol is fixed at 1e-1.
    "test_resnet50",  # max error 1.4
    # AtenToCortexMPass rejects the attention graph: "unsupported param type,
    # call_function" on a bias that is a computed node rather than a parameter.
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
    )


CORTEX_M_TEST_FLOW = _create_cortex_m_flow()
