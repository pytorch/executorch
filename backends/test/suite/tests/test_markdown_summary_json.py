import json
import os
import tempfile
import unittest

from ..generate_markdown_summary_json import aggregate_results


def _test(name: str, outcome: str, results: list[str], longrepr: str = "") -> dict:
    subtests = [
        {
            "Test ID": f"{name}[flow1]",
            "Test Case": name,
            "Subtest": i,
            "Flow": "flow1",
            "Result": result,
            "Result Detail": "Output Mismatch" if result == "Fail" else "",
        }
        for i, result in enumerate(results)
    ]
    test = {
        "nodeid": f"test_file.py::{name}[flow1]",
        "outcome": outcome,
        "metadata": {"subtests": subtests},
    }
    if longrepr:
        test["call"] = {"longrepr": longrepr}
    return test


class AggregateResultsTest(unittest.TestCase):
    def _aggregate(self, tests: list[dict]):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            with open(path, "w") as f:
                json.dump({"tests": tests}, f)
            return aggregate_results(path)

    def test_expected_failures_are_not_failures(self):
        summary = self._aggregate(
            [
                _test("test_listed", "xfailed", ["Pass", "Fail"]),
                _test("test_broken", "failed", ["Fail"]),
            ]
        )
        self.assertEqual(summary.counts.passes, 1)
        self.assertEqual(summary.counts.fails, 1)
        self.assertEqual(summary.counts.expected_fails, 1)
        self.assertEqual(
            [r["Test ID"] for r in summary.failed_tests], ["test_broken[flow1]"]
        )
        self.assertEqual(
            [r["Test ID"] for r in summary.expected_failures], ["test_listed[flow1]"]
        )
        self.assertEqual(summary.counts.by_detail, {"Output Mismatch": 1})

    def test_strict_unexpected_passes_are_listed(self):
        summary = self._aggregate(
            [
                _test(
                    "test_serial",
                    "failed",
                    ["Pass"],
                    "[XPASS(strict)] Expected to fail by flow1 xfail_patterns",
                ),
                # Under pytest-xdist the first line names the worker.
                _test(
                    "test_xdist",
                    "failed",
                    ["Pass"],
                    "[gw0] linux -- Python 3.11 /usr/bin/python\n"
                    "[XPASS(strict)] Expected to fail by flow1 xfail_patterns",
                ),
                _test("test_plain_pass", "passed", ["Pass"]),
            ]
        )
        self.assertEqual(
            summary.unexpected_passes, ["test_serial[flow1]", "test_xdist[flow1]"]
        )
        self.assertEqual(summary.counts.passes, 3)
        self.assertEqual(summary.failed_tests, [])
