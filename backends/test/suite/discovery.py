# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import unittest

from dataclasses import dataclass
from types import ModuleType
from typing import Pattern

from executorch.backends.test.suite.flow import TestFlow

#
# This file contains logic related to test discovery and filtering.
#


@dataclass
class TestFilter:
    """A set of filters for test discovery."""

    backends: set[str] | None
    """ The set of backends to include. If None, all backends are included. """

    flows: set[str] | None
    """ The set of test flows to include. If None, all backends are included. """

    name_regex: Pattern[str] | None
    """ A regular expression to filter test names. If None, all tests are included. """


def discover_tests(
    root_module: ModuleType, test_filter: TestFilter
) -> unittest.TestSuite:
    # Collect all tests using the unittest discovery mechanism then filter down.

    # Find the file system path corresponding to the root module.
    module_file = root_module.__file__
    if module_file is None:
        raise RuntimeError(f"Module {root_module} has no __file__ attribute")

    loader = unittest.TestLoader()
    module_dir = os.path.dirname(module_file)
    suite = loader.discover(module_dir)

    return _filter_tests(suite, test_filter)


def _filter_tests(
    suite: unittest.TestSuite, test_filter: TestFilter
) -> unittest.TestSuite:
    # Recursively traverse the test suite and add them to the filtered set.
    filtered_suite = unittest.TestSuite()

    for child in suite:
        if isinstance(child, unittest.TestSuite):
            filtered_suite.addTest(_filter_tests(child, test_filter))
        elif isinstance(child, unittest.TestCase):
            if _is_test_enabled(child, test_filter):
                filtered_suite.addTest(child)
        else:
            raise RuntimeError(f"Unexpected test type: {type(child)}")

    return filtered_suite


def _is_test_enabled(test_case: unittest.TestCase, test_filter: TestFilter) -> bool:
    test_method = getattr(test_case, test_case._testMethodName)

    # Handle import / discovery failures - leave them enabled to report nicely at the
    # top level. There might be a better way to do this. Internally, unittest seems to
    # replace it with a stub method to report the failure.
    if "testFailure" in str(test_method):
        print(f"Warning: Test {test_case._testMethodName} failed to import.")
        return True

    if not hasattr(test_method, "_flow"):
        raise RuntimeError(
            f"Test missing flow: {test_case._testMethodName} {test_method}"
        )

    flow: TestFlow = test_method._flow

    if test_filter.backends is not None and flow.backend not in test_filter.backends:
        return False

    if test_filter.flows is not None and flow.name not in test_filter.flows:
        return False

    if test_filter.name_regex is not None and not test_filter.name_regex.search(
        test_case.id()
    ):
        return False

    return True
