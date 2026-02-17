# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.apple.mlx.test.tester import MLXTester
from executorch.backends.test.suite.flow import TestFlow

MLX_TEST_FLOW = TestFlow(
    name="mlx",
    backend="mlx",
    tester_factory=MLXTester,
)
