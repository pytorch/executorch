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
from typing import Any, Callable, Tuple

import torch
from executorch.backends.test.harness import Tester

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


ALL_TEST_FLOWS = []

if is_backend_enabled("xnnpack"):
    from executorch.backends.xnnpack.test.tester import Tester as XnnpackTester

    XNNPACK_TEST_FLOW = ("xnnpack", XnnpackTester)
    ALL_TEST_FLOWS.append(XNNPACK_TEST_FLOW)

if is_backend_enabled("coreml"):
    from executorch.backends.apple.coreml.test.tester import CoreMLTester

    COREML_TEST_FLOW = ("coreml", CoreMLTester)
    ALL_TEST_FLOWS.append(COREML_TEST_FLOW)


DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
    torch.int32,
    torch.uint32,
    torch.int64,
    torch.uint64,
    torch.float16,
    torch.float32,
    torch.float64,
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
    for flow_name, tester_factory in ALL_TEST_FLOWS:
        _create_test_for_backend(cls, test_func, flow_name, tester_factory)
    delattr(cls, test_name)


def _make_wrapped_test(test_func, *args, **kwargs):
    def wrapped_test(self):
        test_func(self, *args, **kwargs)

    return wrapped_test


def _make_wrapped_dtype_test(test_func, dtype, tester_factory):
    def wrapped_test(self):
        test_func(self, dtype, tester_factory)

    return wrapped_test


def _create_test_for_backend(
    cls,
    test_func: Callable,
    flow_name: str,
    tester_factory: Callable[[torch.nn.Module, Tuple[Any]], Tester],
):
    test_type = getattr(test_func, "test_type", TestType.STANDARD)

    if test_type == TestType.STANDARD:
        wrapped_test = _make_wrapped_test(test_func, tester_factory)
        test_name = f"{test_func.__name__}_{flow_name}"
        setattr(cls, test_name, wrapped_test)
    elif test_type == TestType.DTYPE:
        for dtype in DTYPES:
            # wrapped_test = _make_wrapped_dtype_test(test_func, dtype, tester_factory)
            wrapped_test = _make_wrapped_test(test_func, dtype, tester_factory)
            dtype_name = str(dtype)[6:]  # strip "torch."
            test_name = f"{test_func.__name__}_{dtype_name}_{flow_name}"
            setattr(cls, test_name, wrapped_test)
    else:
        raise NotImplementedError(f"Unknown test type {test_type}.")


class OperatorTest(unittest.TestCase):
    def _test_op(self, model, inputs, tester_factory):
        tester = (
            tester_factory(
                model,
                inputs,
            )
            .export()
            .to_edge_transform_and_lower()
        )

        is_delegated = any(
            n.target == torch._higher_order_ops.executorch_call_delegate
            for n in tester.stages[tester.cur].graph_module.graph.nodes
            if n.op == "call_function"
        )

        # Only run the runtime test if the op was delegated.
        if is_delegated:
            (tester.to_executorch().serialize().run_method_and_compare_outputs())
