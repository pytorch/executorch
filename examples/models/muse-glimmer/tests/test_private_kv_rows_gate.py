# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "private_kv_rows_checker", Path(__file__).with_name("check_private_kv_rows.py")
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PrivateKVRowsPerformanceGateTest(unittest.TestCase):
    def test_boundary_and_improvements_qualify(self):
        checker.require_performance(
            {
                128: {
                    "same_work_timings": {
                        1: {"throughput_ratio": 0.98},
                        64: {"throughput_ratio": 1.2},
                    }
                }
            }
        )

    def test_measured_regression_blocks_even_without_flag(self):
        with self.assertRaisesRegex(RuntimeError, "d256/c4"):
            checker.require_performance(
                {256: {"same_work_timings": {4: {"throughput_ratio": 0.979}}}}
            )

    def test_one_bad_concurrency_blocks_the_complete_gate(self):
        with self.assertRaisesRegex(RuntimeError, "d128/c64"):
            checker.require_performance(
                {
                    128: {
                        "same_work_timings": {
                            1: {"throughput_ratio": 1.01},
                            64: {"throughput_ratio": 0.7},
                        }
                    },
                    256: {"same_work_timings": {64: {"throughput_ratio": 1.1}}},
                }
            )


if __name__ == "__main__":
    unittest.main()
