# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.test.common import parametrize

from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase

YOLO = pytest.importorskip(
    "ultralytics",
    reason="ultralytics is optional; install it locally to run YOLO tests.",
).YOLO


ops_before_transforms: dict[str, int] = {}
ops_after_transforms: dict[str, int] = {}


WEIGHTS = "yolo11n.pt"
yolo = YOLO(WEIGHTS)
pt_model = yolo.model.eval()

test_cases = {
    "yolo11n": McuTestCase(
        model=pt_model,
        example_inputs=lambda: (
            torch.randn(1, 3, 640, 640).to(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_yolo11(test_case):
    """This model currently does not lower in the cortex-m backend, this test is to track development progress."""
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
    )
