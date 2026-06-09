# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.webgpu.test.tester import WebGPUTester


def _create_webgpu_flow() -> TestFlow:
    return TestFlow(
        "webgpu",
        backend="webgpu",
        tester_factory=WebGPUTester,
        skip_patterns=[
            "float16",
            "float64",  # Not supported in swiftshader
            # WebGPU add is elementwise-only; broadcasting add.Tensor unsupported.
            "bcast_first",
            "bcast_second",
            "hardswish",
            "lstm_batch_sizes",
            "upsample_nearest2d",
            # torchvision models with broadcasting adds; resnet50 covers wide.
            "mobilenet_v3_small",
            "shufflenet_v2_x1_0",
            "resnet50",
            "vit_b_16",
            "swin_v2_t",
            "convnext_small",
        ],
    )


WEBGPU_TEST_FLOW = _create_webgpu_flow()
