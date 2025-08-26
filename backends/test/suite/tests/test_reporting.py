import unittest

from csv import DictReader
from io import StringIO

import torch

from executorch.exir import to_edge

from ..reporting import (
    count_ops,
    generate_csv_report,
    RunSummary,
    TestCaseSummary,
    TestResult,
    TestSessionState,
)

# Test data for simulated test results.
TEST_CASE_SUMMARIES = [
    TestCaseSummary(
        backend="backend1",
        base_name="test1",
        flow="flow1",
        name="test1_backend1_flow1",
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
        params={"use_dynamic_shapes": True},
        result=TestResult.EXPORT_FAIL,
        error=None,
        tensor_error_statistics=[],
    ),
]


class Reporting(unittest.TestCase):
    def test_csv_report_simple(self):
        # Verify the format of a simple CSV run report.
        session_state = TestSessionState()
        session_state.test_case_summaries.extend(TEST_CASE_SUMMARIES)
        run_summary = RunSummary.from_session(session_state)

        strio = StringIO()
        generate_csv_report(run_summary, strio)

        # Attempt to deserialize and validate the CSV report.
        report = DictReader(StringIO(strio.getvalue()))
        records = list(report)
        self.assertEqual(len(records), 4)

        # Validate first record: test1, backend1, SUCCESS
        self.assertEqual(records[0]["Test ID"], "test1_backend1_flow1")
        self.assertEqual(records[0]["Test Case"], "test1")
        self.assertEqual(records[0]["Backend"], "backend1")
        self.assertEqual(records[0]["Flow"], "flow1")
        self.assertEqual(records[0]["Result"], "Success (Delegated)")
        self.assertEqual(records[0]["Dtype"], "")
        self.assertEqual(records[0]["Use_dynamic_shapes"], "")

        # Validate second record: test1, backend2, LOWER_FAIL
        self.assertEqual(records[1]["Test ID"], "test1_backend2_flow1")
        self.assertEqual(records[1]["Test Case"], "test1")
        self.assertEqual(records[1]["Backend"], "backend2")
        self.assertEqual(records[1]["Flow"], "flow1")
        self.assertEqual(records[1]["Result"], "Fail (Lowering)")
        self.assertEqual(records[1]["Dtype"], "")
        self.assertEqual(records[1]["Use_dynamic_shapes"], "")

        # Validate third record: test2, backend1, SUCCESS_UNDELEGATED with dtype param
        self.assertEqual(records[2]["Test ID"], "test2_backend1_flow1")
        self.assertEqual(records[2]["Test Case"], "test2")
        self.assertEqual(records[2]["Backend"], "backend1")
        self.assertEqual(records[2]["Flow"], "flow1")
        self.assertEqual(records[2]["Result"], "Success (Undelegated)")
        self.assertEqual(records[2]["Dtype"], str(torch.float32))
        self.assertEqual(records[2]["Use_dynamic_shapes"], "")

        # Validate fourth record: test2, backend2, EXPORT_FAIL with use_dynamic_shapes param
        self.assertEqual(records[3]["Test ID"], "test2_backend2_flow1")
        self.assertEqual(records[3]["Test Case"], "test2")
        self.assertEqual(records[3]["Backend"], "backend2")
        self.assertEqual(records[3]["Flow"], "flow1")
        self.assertEqual(records[3]["Result"], "Fail (Export)")
        self.assertEqual(records[3]["Dtype"], "")
        self.assertEqual(records[3]["Use_dynamic_shapes"], "True")

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
