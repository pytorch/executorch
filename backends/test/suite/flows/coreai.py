# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from executorch.backends.apple.coreai.test.tester import CoreAIQuantize, CoreAITester
from executorch.backends.test.suite.flow import TestFlow

COREAI_SKIP_PATTERNS = [
    # These include an accumulate=False write to the same position twice, which
    # PyTorch documents as undefined. Core AI keeps the first write and CPU the
    # last, so the reference comparison fails.
    "test_index_put_two_indices",
    "test_index_put_in_place_two_indices",
]

COREAI_TEST_FLOWS = [
    TestFlow(
        name="coreai_fp32",
        backend="coreai",
        tester_factory=CoreAITester,
        skip_patterns=COREAI_SKIP_PATTERNS,
    ),
    TestFlow(
        name="coreai_fp16",
        backend="coreai",
        tester_factory=partial(CoreAITester, fp16=True),
        skip_patterns=[
            *COREAI_SKIP_PATTERNS,
            # PyTorch CPU avg_pool3d has no float16 reference implementation.
            "test_avgpool3d_",
            # The randomly initialized model overflows float16 (fp32 activations
            # reach ~5e11), so the CPU reference is itself all NaN.
            "test_inception_v3",
        ],
    ),
    *[
        TestFlow(
            name=f"coreai_{mode}",
            backend="coreai",
            tester_factory=CoreAITester,
            quantize=True,
            quantize_stage_factory=partial(
                CoreAIQuantize, mode, skip_if_unquantizable=True
            ),
            skip_patterns=COREAI_SKIP_PATTERNS,
        )
        for mode in ("w8a8", "w8")
    ],
]
