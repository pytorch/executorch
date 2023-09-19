# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: F401
import functools
import inspect
import os
import random
import unittest
from typing import Callable, Dict, Optional, Tuple, Type

import executorch.exir as exir

import executorch.exir.control_flow as control_flow

# @manual=//executorch/extension/pytree:pybindings
import executorch.extension.pytree as pytree

import torch
from executorch.bundled_program.config import BundledConfig

from executorch.bundled_program.core import create_bundled_program
from executorch.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from executorch.bundled_program.tests.common import get_common_program, SampleModel

kernel_mode = None  # either aten mode or lean mode
try:
    # pyre-ignore[21]
    from executorch.extension.pybindings.portable_lib import (
        _load_bundled_program_from_buffer,
        _load_for_executorch_from_buffer,
        _load_for_executorch_from_bundled_program,
    )

    kernel_mode = "lean"
except ImportError as e:
    print(e)
    pass

try:
    # pyre-ignore[21]
    from executorch.extension.pybindings.aten_lib import (
        _load_bundled_program_from_buffer,
        _load_for_executorch_from_buffer,
        _load_for_executorch_from_bundled_program,
    )

    assert kernel_mode is None
    kernel_mode = "aten"
except ImportError as e:
    print(e)
    pass

assert kernel_mode is not None


class BundledProgramE2ETest(unittest.TestCase):
    def test_sample_model_e2e(self):
        program, bundled_config = get_common_program()
        eager_model = SampleModel()

        bundled_program = create_bundled_program(program, bundled_config)

        bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )

        executorch_bundled_program = _load_bundled_program_from_buffer(
            bundled_program_buffer
        )

        executorch_module = _load_for_executorch_from_bundled_program(
            executorch_bundled_program
        )

        for method_name in eager_model.method_names:
            executorch_module.load_bundled_input(
                executorch_bundled_program,
                method_name,
                0,
            )
            executorch_module.plan_execute(method_name)
            executorch_module.verify_result_with_bundled_expected_output(
                executorch_bundled_program,
                method_name,
                0,
            )
