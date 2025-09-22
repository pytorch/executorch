# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
import os
import unittest
from typing import Any, Callable

import torch
from executorch.backends.test.suite import get_test_flows
from executorch.backends.test.suite.context import get_active_test_context, TestContext
from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.test.suite.reporting import log_test_summary
from executorch.backends.test.suite.runner import run_test


DTYPES: list[torch.dtype] = [
    torch.float16,
    torch.float32,
]


class ModelTest(unittest.TestCase):
    pass


class TestCaseShim:
    def __init__(self, test_runner):
        self._test_runner = test_runner

    def _test_op(self, model, args, flow, generate_random_test_inputs=True):
        self._test_runner.lower_and_run_model(model, args)


def wrap_test(original_func, test_type):
    def wrapped_func(test_runner):
        shim = TestCaseShim(test_runner)
        original_func(shim, test_runner._flow)

    return wrapped_func


def model_test_cls(cls):
    parent_module = sys.modules[cls.__module__]

    for func_name in dir(cls):
        if func_name.startswith("test"):
            original_func = getattr(cls, func_name)
            test_type = getattr(original_func, "test_type", TestType.STANDARD)
            wrapped_func = wrap_test(original_func, test_type)
            setattr(parent_module, func_name, wrapped_func)

    return None


def model_test_params(
    supports_dynamic_shapes: bool = True,
    dtypes: list[torch.dtype] | None = None,
) -> Callable:
    """Optional parameter decorator for model tests. Specifies test pararameters. Only valid with a class decorated by model_test_cls."""

    def inner_decorator(func: Callable) -> Callable:
        func.supports_dynamic_shapes = supports_dynamic_shapes  # type: ignore

        if dtypes is not None:
            func.dtypes = dtypes  # type: ignore

        return func

    return inner_decorator
