# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from pathlib import Path
from unittest import mock

from . import performance


class TestPerformanceTestId(unittest.TestCase):
    def _test_id(self, current_test: str) -> str:
        with mock.patch.dict(os.environ, {"PYTEST_CURRENT_TEST": current_test}):
            return performance._current_pytest_test_id()

    def test_includes_class_hierarchy(self) -> None:
        path = "backends/xnnpack/test/models/example.py"
        first = self._test_id(f"{path}::TestFirst::test_model (call)")
        second = self._test_id(f"{path}::TestSecond::test_model (call)")

        self.assertNotEqual(first, second)
        self.assertIn(".testfirst.test_model.", first)
        self.assertIn(".testsecond.test_model.", second)

    def test_slug_collisions_have_distinct_digests(self) -> None:
        path = "backends/xnnpack/test/models/example.py::TestModel::test_model"
        slash = self._test_id(f"{path}[a/b] (call)")
        space = self._test_id(f"{path}[a b] (call)")

        self.assertNotEqual(slash, space)
        self.assertEqual(slash.rsplit(".", 1)[0], space.rsplit(".", 1)[0])

    def test_derivation_is_stable(self) -> None:
        current_test = (
            "backends/xnnpack/test/models/example.py::TestModel::test_model[value] "
            "(call)"
        )
        self.assertEqual(self._test_id(current_test), self._test_id(current_test))

    def test_removes_pytest_phase_suffix(self) -> None:
        node_id = "backends/xnnpack/test/models/example.py::TestModel::test_model"
        self.assertEqual(self._test_id(node_id), self._test_id(f"{node_id} (call)"))

    def test_default_results_path_uses_test_module(self) -> None:
        test_id = self._test_id(
            "backends/xnnpack/test/models/resnet.py::TestResNet18::"
            "test_fp32_resnet18 (call)"
        )
        expected = (
            Path(performance.__file__).resolve().parents[1]
            / "models"
            / "resnet_pytest_perf_results.json"
        )
        self.assertEqual(performance._default_results_path(test_id), expected)


class TestPerformanceThreadCount(unittest.TestCase):
    def test_configured_thread_count_is_used(self) -> None:
        record = {"timing_runtime": {"runtime_key": "runtime"}}
        with (
            mock.patch.dict(
                os.environ,
                {
                    performance.PERF_ENABLE_ENV: "1",
                    performance.PERF_THREADS_ENV: "4",
                },
                clear=True,
            ),
            mock.patch.object(
                performance, "_current_pytest_test_id", return_value="test-id"
            ),
            mock.patch.object(
                performance, "_measure_latency", return_value=record
            ) as measure,
            mock.patch.object(
                performance, "_load_results", return_value={"entries": {}}
            ),
        ):
            performance.maybe_run_performance_test(
                serialized_buffer=b"pte",
                inputs=(),
                results_path="results.json",
            )

        self.assertEqual(measure.call_args.kwargs["thread_count"], 4)

    def test_thread_count_separates_runtime_keys(self) -> None:
        native_path = Path("_portable_lib.so")
        host = {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_id": "test-cpu",
            "sme2_available": False,
        }

        single_threaded = performance._runtime_key(native_path, host, 1)
        multi_threaded = performance._runtime_key(native_path, host, 4)

        self.assertNotEqual(single_threaded, multi_threaded)
        self.assertIn("threads1", single_threaded)
        self.assertIn("threads4", multi_threaded)


if __name__ == "__main__":
    unittest.main()
