# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: F401
import unittest

import torch
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from executorch.devtools.bundled_program.util.test_util import (
    get_common_executorch_program,
    SampleModel,
)
from executorch.exir import to_edge

kernel_mode = None  # either aten mode or lean mode
try:
    from executorch.extension.pybindings.portable_lib import (
        _load_bundled_program_from_buffer,
    )

    kernel_mode = "lean"
except ImportError as e:
    print(e)
    pass

try:
    from executorch.extension.pybindings.aten_lib import (  # @manual=//executorch/extension/pybindings:aten_lib
        _load_bundled_program_from_buffer,
    )

    assert kernel_mode is None
    kernel_mode = "aten"
except ImportError as e:
    print(e)
    pass

assert kernel_mode is not None


class BundledProgramE2ETest(unittest.TestCase):
    def test_none_input_e2e(self):
        class OptionalInput(torch.nn.Module):
            def forward(self, x, optional, y):
                return x - y if optional is None else x + optional + y

        model = OptionalInput()
        inputs = (torch.ones(2), None, torch.full((2,), 3.0))
        program = to_edge(torch.export.export(model, inputs)).to_executorch()
        bundle = BundledProgram(
            program,
            [MethodTestSuite("forward", [MethodTestCase(inputs, (model(*inputs),))])],
        )
        runtime = _load_bundled_program_from_buffer(
            serialize_from_bundled_program_to_flatbuffer(bundle)
        )
        runtime.verify_result_with_bundled_expected_output("forward", 0)

    def test_sample_model_e2e(self):
        executorch_program, method_test_suites = get_common_executorch_program()
        eager_model = SampleModel()

        bundled_program = BundledProgram(executorch_program, method_test_suites)

        bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )

        executorch_bundled_program = _load_bundled_program_from_buffer(
            bundled_program_buffer
        )

        for method_name in eager_model.method_names:
            executorch_bundled_program.verify_result_with_bundled_expected_output(
                method_name,
                0,
            )
