# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import unittest

from types import ModuleType

from executorch.backends.test.suite.flow import TestFlow

#
# This file contains logic related to test discovery and filtering.
#


def discover_tests(
    root_module: ModuleType, backends: set[str] | None
) -> unittest.TestSuite:
    # Collect all tests using the unittest discovery mechanism then filter down.

    # Find the file system path corresponding to the root module.
    module_file = root_module.__file__
    if module_file is None:
        raise RuntimeError(f"Module {root_module} has no __file__ attribute")

    loader = unittest.TestLoader()
    module_dir = os.path.dirname(module_file)
    suite = loader.discover(module_dir)

    return _filter_tests(suite, backends)


def _filter_tests(
    suite: unittest.TestSuite, backends: set[str] | None
) -> unittest.TestSuite:
    # Recursively traverse the test suite and add them to the filtered set.
    filtered_suite = unittest.TestSuite()

    for child in suite:
        if isinstance(child, unittest.TestSuite):
            filtered_suite.addTest(_filter_tests(child, backends))
        elif isinstance(child, unittest.TestCase):
            if _is_test_enabled(child, backends):
                filtered_suite.addTest(child)
        else:
            raise RuntimeError(f"Unexpected test type: {type(child)}")

    return filtered_suite


def _is_test_enabled(test_case: unittest.TestCase, backends: set[str] | None) -> bool:
    test_method = getattr(test_case, test_case._testMethodName)

    if backends is not None:
        flow: TestFlow = getattr(test_method, "_flow")
        return flow.backend in backends
    else:
        return True
