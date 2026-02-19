# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cuda.test.tester import CudaTester
from executorch.backends.test.suite.flow import TestFlow


def _create_cuda_flow(name: str = "cuda") -> TestFlow:
    """Create a test flow for the CUDA backend.

    The CUDA backend saves data externally (.so and weights blob in .ptd file).
    The test harness serialize stage has been updated to support loading external
    data via the data_map_buffer parameter of _load_for_executorch_from_buffer.
    """
    return TestFlow(
        name,
        backend="cuda",
        tester_factory=CudaTester,
        quantize=False,
        # Skip tests that cause SIGABRT crashes in CUDA runtime
        # test_mean_output_dtype: float64 output dtype causes crash in AOTI compiled code
        skip_patterns=["test_mean_output_dtype"],
    )


CUDA_TEST_FLOW = _create_cuda_flow("cuda")
