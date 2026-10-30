# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from executorch.backends.test.harness.tester import Tester as BackendTester


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_mismatch_reports_statistics(dtype):
    model = torch.tensor([1.0, 2.0, 3.0]).to(dtype)
    reference = torch.tensor([1.0, 2.0, 4.0]).to(dtype)

    with pytest.raises(
        AssertionError, match="Output 0 does not match reference output"
    ) as error:
        BackendTester._assert_outputs_equal([model], [reference], atol=0, rtol=0)

    message = str(error.value)
    assert f"dtype: {dtype}" in message
    assert "Median: 2.0, 2.0" in message
    assert "Max: 3.0, 4.0" in message
    assert "Min: 1.0, 1.0" in message
