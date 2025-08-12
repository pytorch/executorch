import csv
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from typing import TextIO

from executorch.backends.test.harness.error_statistics import ErrorStatistics


class TestResult(IntEnum):
    """Represents the result of a test case run, indicating success or a specific failure reason."""

    SUCCESS = 0
    """ The test succeeded with the backend delegate part or all of the graph. """

    SUCCESS_UNDELEGATED = 1
    """ The test succeeded without the backend delegating anything. """

    EAGER_FAIL = 2
    """ The test failed due to the model failing to run in eager mode. """

    QUANTIZE_FAIL = 3
    """ The test failed due to the quantization stage failing. """

    EXPORT_FAIL = 4
    """ The test failed due to the model failing to export. """

    LOWER_FAIL = 5
    """ The test failed due to a failure in partitioning or lowering. """

    PTE_LOAD_FAIL = 6
    """ The test failed due to the resulting PTE failing to load. """

    PTE_RUN_FAIL = 7
    """ The test failed due to the resulting PTE failing to run. """

    OUTPUT_MISMATCH_FAIL = 8
    """ The test failed due to a mismatch between runtime and reference outputs. """

    UNKNOWN_FAIL = 9
    """ The test failed in an unknown or unexpected manner. """

    def is_success(self):
        return self in {TestResult.SUCCESS, TestResult.SUCCESS_UNDELEGATED}

    def is_non_backend_failure(self):
        return self in {TestResult.EAGER_FAIL, TestResult.EAGER_FAIL}

    def is_backend_failure(self):
        return not self.is_success() and not self.is_non_backend_failure()

    def display_name(self):
        if self == TestResult.SUCCESS:
            return "Success (Delegated)"
        elif self == TestResult.SUCCESS_UNDELEGATED:
            return "Success (Undelegated)"
        elif self == TestResult.EAGER_FAIL:
            return "Fail (Eager)"
        elif self == TestResult.QUANTIZE_FAIL:
            return "Fail (Quantize)"
        elif self == TestResult.EXPORT_FAIL:
            return "Fail (Export)"
        elif self == TestResult.LOWER_FAIL:
            return "Fail (Lowering)"
        elif self == TestResult.PTE_LOAD_FAIL:
            return "Fail (PTE Load)"
        elif self == TestResult.PTE_RUN_FAIL:
            return "Fail (PTE Run)"
        elif self == TestResult.OUTPUT_MISMATCH_FAIL:
            return "Fail (Output Mismatch)"
        elif self == TestResult.UNKNOWN_FAIL:
            return "Fail (Other)"
        else:
            raise ValueError(f"Invalid TestResult value: {self}.")


@dataclass
class TestCaseSummary:
    """
    Contains summary results for the execution of a single test case.
    """

    backend: str
    """ The name of the target backend. """

    base_name: str
    """ The base name of the test, not including flow or parameter suffixes. """

    flow: str
    """ The backend-specific flow name. Corresponds to flows registered in backends/test/suite/__init__.py. """

    name: str
    """ The full name of test, including flow and parameter suffixes. """

    params: dict | None
    """ Test-specific parameters, such as dtype. """

    result: TestResult
    """ The top-level result, such as SUCCESS or LOWER_FAIL. """

    error: Exception | None
    """ The Python exception object, if any. """

    tensor_error_statistics: list[ErrorStatistics]
    """ 
    Statistics about the error between the backend and reference outputs. Each element of this list corresponds to
    a single output tensor.
    """


class TestSessionState:
    test_case_summaries: list[TestCaseSummary]

    def __init__(self):
        self.test_case_summaries = []


@dataclass
class RunSummary:
    aggregated_results: dict[TestResult, int]
    num_test_cases: int
    test_case_summaries: list[TestCaseSummary]
    total_failed: int
    total_passed: int
    total_skipped: int

    @classmethod
    def from_session(cls, session: TestSessionState) -> "RunSummary":
        # Total each outcome type.
        aggregated_results = dict(
            sorted(Counter(s.result for s in session.test_case_summaries).items())
        )

        total_failed = 0
        total_passed = 0
        total_skipped = 0

        for k, v in aggregated_results.items():
            if k.is_success():
                total_passed += v
            elif k.is_backend_failure():
                total_failed += v
            else:
                total_skipped += v

        return cls(
            aggregated_results=aggregated_results,
            num_test_cases=len(session.test_case_summaries),
            test_case_summaries=session.test_case_summaries,
            total_failed=total_failed,
            total_passed=total_passed,
            total_skipped=total_skipped,
        )


_active_session: TestSessionState | None = None


def begin_test_session():
    global _active_session

    assert _active_session is None, "A test session is already active."
    _active_session = TestSessionState()


def log_test_summary(summary: TestCaseSummary):
    global _active_session

    if _active_session is not None:
        _active_session.test_case_summaries.append(summary)


def complete_test_session() -> RunSummary:
    global _active_session

    assert _active_session is not None, "No test session is active."
    summary = RunSummary.from_session(_active_session)
    _active_session = None

    return summary


def generate_csv_report(summary: RunSummary, output: TextIO):
    """Write a run summary report to a file in CSV format."""

    field_names = [
        "Test ID",
        "Test Case",
        "Backend",
        "Flow",
        "Result",
    ]

    # Tests can have custom parameters. We'll want to report them here, so we need
    # a list of all unique parameter names.
    param_names = reduce(
        lambda a, b: a.union(b),
        (
            set(s.params.keys())
            for s in summary.test_case_summaries
            if s.params is not None
        ),
        set(),
    )
    field_names += (s.capitalize() for s in param_names)

    # Add tensor error statistic field names for each output index.
    max_outputs = max(
        len(s.tensor_error_statistics) for s in summary.test_case_summaries
    )
    for i in range(max_outputs):
        field_names.extend(
            [
                f"Output {i} Error Max",
                f"Output {i} Error MAE",
                f"Output {i} Error MSD",
                f"Output {i} Error L2",
                f"Output {i} SQNR",
            ]
        )

    writer = csv.DictWriter(output, field_names)
    writer.writeheader()

    for record in summary.test_case_summaries:
        row = {
            "Test ID": record.name,
            "Test Case": record.base_name,
            "Backend": record.backend,
            "Flow": record.flow,
            "Result": record.result.display_name(),
        }
        if record.params is not None:
            row.update({k.capitalize(): v for k, v in record.params.items()})

        for output_idx, error_stats in enumerate(record.tensor_error_statistics):
            row[f"Output {output_idx} Error Max"] = error_stats.error_max
            row[f"Output {output_idx} Error MAE"] = error_stats.error_mae
            row[f"Output {output_idx} Error MSD"] = error_stats.error_msd
            row[f"Output {output_idx} Error L2"] = error_stats.error_l2_norm
            row[f"Output {output_idx} SQNR"] = error_stats.sqnr

        writer.writerow(row)
