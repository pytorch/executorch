# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import argparse
import os
import sys
import unittest

from executorch.backends.samsung.test.utils.utils import TestConfig


TESTS_SEARCH_DIRS = ["ops", "models"]
current_dir = os.path.dirname(os.path.abspath(__file__))


def setup_env_with_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chipset",
        default="E9955",
        help="Samsung chipset, i.e. E9955, E9965, etc",
        type=str,
    )
    parser.add_argument(
        "--host",
        help="Host ip address with device connecting",
        type=str,
    )
    args = parser.parse_args()

    TestConfig.host_ip = args.host
    TestConfig.chipset = args.chipset


if __name__ == "__main__":
    setup_env_with_args()
    test_suite = unittest.TestSuite()

    for test_search_dir in TESTS_SEARCH_DIRS:
        tests = unittest.TestLoader().discover(
            start_dir=os.path.join(f"{current_dir}/../", test_search_dir),
            pattern="test*.py",
            top_level_dir=None,
        )
        test_suite.addTest(tests)

    test_runner = unittest.TextTestRunner()
    result = test_runner.run(test_suite)

    if not result.wasSuccessful():
        print("----------------------------------------------------------------------")
        for fail_case in result.failures:
            print(f"  {fail_case[0]}")
        sys.exit(1)
