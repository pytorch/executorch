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


def load_tests(loader, suite, pattern):
    package_dir = os.path.dirname(__file__)
    discovered_suite = loader.discover(
        start_dir=package_dir, pattern=pattern or "test_*.py"
    )
    suite.addTests(discovered_suite)
    return suite


def _create_test(
    cls,
    test_func: Callable,
    flow: TestFlow,
    dtype: torch.dtype,
    use_dynamic_shapes: bool,
):
    dtype_name = str(dtype)[6:]  # strip "torch."
    test_name = f"{test_func.__name__}_{flow.name}_{dtype_name}"
    if use_dynamic_shapes:
        test_name += "_dynamic_shape"

    def wrapped_test(self):
        params = {
            "dtype": dtype,
            "use_dynamic_shapes": use_dynamic_shapes,
        }
        with TestContext(test_name, test_func.__name__, flow.name, params):
            test_func(self, flow, dtype, use_dynamic_shapes)

    wrapped_test._name = test_func.__name__  # type: ignore
    wrapped_test._flow = flow  # type: ignore

    setattr(cls, test_name, wrapped_test)


# Expand a test into variants for each registered flow.
def _expand_test(cls, test_name: str) -> None:
    test_func = getattr(cls, test_name)
    supports_dynamic_shapes = getattr(test_func, "supports_dynamic_shapes", True)
    dynamic_shape_values = [True, False] if supports_dynamic_shapes else [False]
    dtypes = getattr(test_func, "dtypes", DTYPES)

    for flow, dtype, use_dynamic_shapes in itertools.product(
        get_test_flows().values(), dtypes, dynamic_shape_values
    ):
        _create_test(cls, test_func, flow, dtype, use_dynamic_shapes)
    delattr(cls, test_name)


def model_test_cls(cls) -> Callable | None:
    """Decorator for model tests. Handles generating test variants for each test flow and configuration."""
    for key in dir(cls):
        if key.startswith("test_"):
            _expand_test(cls, key)
    return cls


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


def run_model_test(
    model: torch.nn.Module,
    inputs: tuple[Any],
    flow: TestFlow,
    dtype: torch.dtype,
    dynamic_shapes: Any | None,
):
    model = model.to(dtype)
    context = get_active_test_context()

    # This should be set in the wrapped test. See _create_test above.
    assert context is not None, "Missing test context."

    run_summary = run_test(
        model,
        inputs,
        flow,
        context.test_name,
        context.test_base_name,
        0,  # subtest_index - currently unused for model tests
        context.params,
        dynamic_shapes=dynamic_shapes,
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
