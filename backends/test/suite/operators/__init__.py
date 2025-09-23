# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import sys
import unittest

from enum import Enum

import pytest
import torch


def load_tests(loader, suite, pattern):
    package_dir = os.path.dirname(__file__)
    discovered_suite = loader.discover(
        start_dir=package_dir, pattern=pattern or "test_*.py"
    )
    suite.addTests(discovered_suite)
    return suite


DTYPES = [
    # torch.int8,
    # torch.uint8,
    # torch.int16,
    # torch.uint16,
    # torch.int32,
    # torch.uint32,
    # torch.int64,
    # torch.uint64,
    # torch.float16,
    torch.float32,
    # torch.float64,
]

FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
]


# The type of test function. This controls the test generation and expected signature.
# Standard tests are run, as is. Dtype tests get a variant generated for each dtype and
# take an additional dtype parameter.
class TestType(Enum):
    STANDARD = 1
    DTYPE = 2


# Function annotation for dtype tests. This instructs the test framework to run the test
# for each supported dtype and to pass dtype as a test parameter.
def dtype_test(func):
    func.test_type = TestType.DTYPE
    return func


class OperatorTest(unittest.TestCase):
    pass


class TestCaseShim:
    def __init__(self, test_runner):
        self._test_runner = test_runner

    def _test_op(self, model, args, flow, generate_random_test_inputs=True):
        self._test_runner.lower_and_run_model(model, args)


def wrap_test(original_func, test_type):
    if test_type == TestType.STANDARD:

        def wrapped_func(test_runner):
            shim = TestCaseShim(test_runner)
            original_func(shim, test_runner._flow)

        return wrapped_func
    elif test_type == TestType.DTYPE:

        @pytest.mark.parametrize(
            "dtype", [torch.float16, torch.float32], ids=lambda s: str(s)[6:]
        )
        def wrapped_func(test_runner, dtype):
            shim = TestCaseShim(test_runner)
            original_func(shim, test_runner._flow, dtype)

        return wrapped_func
    else:
        raise ValueError()


def operator_test(cls):
    parent_module = sys.modules[cls.__module__]

    for func_name in dir(cls):
        if func_name.startswith("test"):
            original_func = getattr(cls, func_name)
            test_type = getattr(original_func, "test_type", TestType.STANDARD)
            wrapped_func = wrap_test(original_func, test_type)
            setattr(parent_module, func_name, wrapped_func)

    return None
