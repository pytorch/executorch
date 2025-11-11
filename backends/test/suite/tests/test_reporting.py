import json
import unittest

from csv import DictReader
from io import StringIO

import torch

from executorch.exir import to_edge

from ..reporting import (
    count_ops,
    RunSummary,
    TestCaseSummary,
    TestResult,
    TestSessionState,
    write_csv_header,
    write_csv_row,
)

# Test data for simulated test results.
TEST_CASE_SUMMARIES = [
    TestCaseSummary(
        backend="backend1",
        base_name="test1",
        flow="flow1",
        name="test1_backend1_flow1",
        subtest_index=0,
        params=None,
        result=TestResult.SUCCESS,
        error=None,
        tensor_error_statistics=[],
    ),
    TestCaseSummary(
        backend="backend2",
        base_name="test1",
        flow="flow1",
        name="test1_backend2_flow1",
        subtest_index=0,
        params=None,
        result=TestResult.LOWER_FAIL,
        error=None,
        tensor_error_statistics=[],
    ),
    TestCaseSummary(
        backend="backend1",
        base_name="test2",
        flow="flow1",
        name="test2_backend1_flow1",
        subtest_index=0,
        params={"dtype": torch.float32},
        result=TestResult.SUCCESS_UNDELEGATED,
        error=None,
        tensor_error_statistics=[],
    ),
    TestCaseSummary(
        backend="backend2",
        base_name="test2",
        flow="flow1",
        name="test2_backend2_flow1",
        subtest_index=0,
        params={"use_dynamic_shapes": True},
        result=TestResult.SKIPPED,
        error=None,
        tensor_error_statistics=[],
    ),
]


class Reporting(unittest.TestCase):
    def test_csv_report_simple(self):
        # Verify the format of a simple CSV run report.
        session_state = TestSessionState(seed=0)
        session_state.test_case_summaries.extend(TEST_CASE_SUMMARIES)
        run_summary = RunSummary.from_session(session_state)

        strio = StringIO()
        write_csv_header(strio)
        for case_summary in run_summary.test_case_summaries:
            write_csv_row(case_summary, strio)

        # Attempt to deserialize and validate the CSV report.
        report = DictReader(StringIO(strio.getvalue()))
        records = list(report)
        self.assertEqual(len(records), 4)

        # Validate first record: test1, backend1, SUCCESS
        self.assertEqual(records[0]["Test ID"], "test1_backend1_flow1")
        self.assertEqual(records[0]["Test Case"], "test1")
        self.assertEqual(records[0]["Flow"], "flow1")
        self.assertEqual(records[0]["Result"], "Pass")
        self.assertEqual(records[0]["Params"], "")

        # Validate second record: test1, backend2, LOWER_FAIL
        self.assertEqual(records[1]["Test ID"], "test1_backend2_flow1")
        self.assertEqual(records[1]["Test Case"], "test1")
        self.assertEqual(records[1]["Flow"], "flow1")
        self.assertEqual(records[1]["Result"], "Fail")
        self.assertEqual(records[1]["Params"], "")

        # Validate third record: test2, backend1, SUCCESS_UNDELEGATED with dtype param
        self.assertEqual(records[2]["Test ID"], "test2_backend1_flow1")
        self.assertEqual(records[2]["Test Case"], "test2")
        self.assertEqual(records[2]["Flow"], "flow1")
        self.assertEqual(records[2]["Result"], "Pass")
        self.assertEqual(records[2]["Params"], json.dumps({"dtype": "torch.float32"}))

        # Validate fourth record: test2, backend2, EXPORT_FAIL with use_dynamic_shapes param
        self.assertEqual(records[3]["Test ID"], "test2_backend2_flow1")
        self.assertEqual(records[3]["Test Case"], "test2")
        self.assertEqual(records[3]["Flow"], "flow1")
        self.assertEqual(records[3]["Result"], "Skip")
        self.assertEqual(
            records[3]["Params"], json.dumps({"use_dynamic_shapes": "True"})
        )

    def test_count_ops(self):
        """
        Verify that the count_ops function correctly counts operator occurances in the edge graph.
        """

        class Model1(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Model2(torch.nn.Module):
            def forward(self, x, y):
                return x + y * y

        args = (torch.randn(2), torch.randn(2))
        ep1 = torch.export.export(Model1(), args)
        ep2 = torch.export.export(Model2(), args)

        ep = to_edge({"forward1": ep1, "forward2": ep2})

        op_counts = count_ops(ep._edge_programs)

        self.assertEqual(
            op_counts,
            {
                "aten::add.Tensor": 2,
                "aten::mul.Tensor": 1,
            },
        )
