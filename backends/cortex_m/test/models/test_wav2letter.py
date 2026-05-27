# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.examples.models.wav2letter.model import Wav2LetterModel


ops_before_transforms: dict[str, int] = {}
ops_after_transforms: dict[str, int] = {}

model = Wav2LetterModel()
pt_model = model.get_eager_model()

test_cases = {
    "wav2letter": McuTestCase(
        model=pt_model,
        example_inputs=lambda: model.get_example_inputs(),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_wav2letter(test_case):
    """This model currently does largely not lower to accelerated kernels due to missing conv1d support, this test is to track development progress."""
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
    )
