# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import logging
import os
import unittest

from enum import Enum
from typing import Callable, Sequence, Sequence

import executorch.backends.test.suite.flow

import torch
from executorch.backends.test.suite.context import get_active_test_context, TestContext
from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.test.suite.reporting import log_test_summary
from executorch.backends.test.suite.runner import run_test, runner_main

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Read enabled backends from the environment variable. Enable all if
# not specified (signalled by None).
def get_enabled_backends():
    et_test_backends = os.environ.get("ET_TEST_ENABLED_BACKENDS")
    if et_test_backends is not None:
        return et_test_backends.split(",")
    else:
        return None


_ENABLED_BACKENDS = get_enabled_backends()


def is_backend_enabled(backend):
    if _ENABLED_BACKENDS is None:
        return True
    else:
        return backend in _ENABLED_BACKENDS


_ALL_TEST_FLOWS: Sequence[TestFlow] | None = None


def get_test_flows() -> Sequence[TestFlow]:
    global _ALL_TEST_FLOWS

    if _ALL_TEST_FLOWS is None:
        _ALL_TEST_FLOWS = [
            f
            for f in executorch.backends.test.suite.flow.all_flows()
            if is_backend_enabled(f.backend)
        ]

    return _ALL_TEST_FLOWS


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


# Class annotation for operator tests. This triggers the test framework to register
# the tests.
def operator_test(cls):
    _create_tests(cls)
    return cls


# Generate test cases for each backend flow.
def _create_tests(cls):
    for key in dir(cls):
        if key.startswith("test_"):
            _expand_test(cls, key)


# Expand a test into variants for each registered flow.
def _expand_test(cls, test_name: str):
    test_func = getattr(cls, test_name)
    for flow in get_test_flows():
        _create_test_for_backend(cls, test_func, flow)
    delattr(cls, test_name)


def _make_wrapped_test(
    test_func: Callable,
    test_name: str,
    flow: TestFlow,
    params: dict | None = None,
):
    def wrapped_test(self):
        with TestContext(test_name, flow.name, params):
            test_kwargs = params or {}
            test_kwargs["tester_factory"] = flow.tester_factory

            test_func(self, **test_kwargs)

    setattr(wrapped_test, "_name", test_name)
    setattr(wrapped_test, "_flow", flow)

    return wrapped_test


def _create_test_for_backend(
    cls,
    test_func: Callable,
    flow: TestFlow,
):
    test_type = getattr(test_func, "test_type", TestType.STANDARD)

    if test_type == TestType.STANDARD:
        wrapped_test = _make_wrapped_test(test_func, test_func.__name__, flow)
        test_name = f"{test_func.__name__}_{flow.name}"
        setattr(cls, test_name, wrapped_test)
    elif test_type == TestType.DTYPE:
        for dtype in DTYPES:
            wrapped_test = _make_wrapped_test(
                test_func,
                test_func.__name__,
                flow,
                {"dtype": dtype},
            )
            dtype_name = str(dtype)[6:]  # strip "torch."
            test_name = f"{test_func.__name__}_{dtype_name}_{flow.name}"
            setattr(cls, test_name, wrapped_test)
    else:
        raise NotImplementedError(f"Unknown test type {test_type}.")


def load_tests(loader, suite, pattern):
    package_dir = os.path.dirname(__file__)
    discovered_suite = loader.discover(
        start_dir=package_dir, pattern=pattern or "test_*.py"
    )
    suite.addTests(discovered_suite)
    return suite


class OperatorTest(unittest.TestCase):
    def _test_op(self, model, inputs, tester_factory):
        context = get_active_test_context()

        # This should be set in the wrapped test. See _make_wrapped_test above.
        assert context is not None, "Missing test context."

        run_summary = run_test(
            model,
            inputs,
            tester_factory,
            context.test_name,
            context.flow_name,
            context.params,
        )

        log_test_summary(run_summary)

        if not run_summary.result.is_success():
            if run_summary.result.is_backend_failure():
                raise RuntimeError("Test failure.") from run_summary.error
            else:
                # Non-backend failure indicates a bad test. Mark as skipped.
                raise unittest.SkipTest(
                    f"Test failed for reasons other than backend failure. Error: {run_summary.error}"
                )


if __name__ == "__main__":
    runner_main()
