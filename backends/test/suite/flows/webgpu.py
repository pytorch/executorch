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
            "hardswish",
            "lstm_batch_sizes",
            "upsample_nearest2d",
        ],
        # WebGPU fails to build the graph for replicate padding ("clamp: int
        # dtype unsupported").
        xfail_patterns=["test_conv3d_padding_replicate"],
    )


WEBGPU_TEST_FLOW = _create_webgpu_flow()
