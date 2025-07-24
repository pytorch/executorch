# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import logging
import os

import executorch.backends.test.suite.flow

from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.test.suite.runner import runner_main

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


_ALL_TEST_FLOWS: dict[str, TestFlow] = {}


def get_test_flows() -> dict[str, TestFlow]:
    global _ALL_TEST_FLOWS

    if not _ALL_TEST_FLOWS:
        _ALL_TEST_FLOWS = {
            name: f
            for name, f in executorch.backends.test.suite.flow.all_flows().items()
            if is_backend_enabled(f.backend)
        }

    return _ALL_TEST_FLOWS


def load_tests(loader, suite, pattern):
    package_dir = os.path.dirname(__file__)
    discovered_suite = loader.discover(
        start_dir=package_dir, pattern=pattern or "test_*.py"
    )
    suite.addTests(discovered_suite)
    return suite


if __name__ == "__main__":
    runner_main()
