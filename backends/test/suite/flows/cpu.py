# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cpu.test.tester import CPUTester
from executorch.backends.test.suite.flow import TestFlow

CPU_FP32_TEST_FLOW = TestFlow("cpu_fp32", backend="cpu", tester_factory=CPUTester)
