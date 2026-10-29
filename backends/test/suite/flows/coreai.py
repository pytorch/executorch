# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from executorch.backends.apple.coreai.test.tester import CoreAIQuantize, CoreAITester
from executorch.backends.test.suite.flow import TestFlow

COREAI_TEST_FLOWS = [
    TestFlow(name="coreai_fp32", backend="coreai", tester_factory=CoreAITester),
    TestFlow(
        name="coreai_fp16",
        backend="coreai",
        tester_factory=partial(CoreAITester, fp16=True),
    ),
    *[
        TestFlow(
            name=f"coreai_{mode}",
            backend="coreai",
            tester_factory=CoreAITester,
            quantize=True,
            quantize_stage_factory=partial(CoreAIQuantize, mode),
        )
        for mode in ("w8a8", "w8", "w4")
    ],
]
