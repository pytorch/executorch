import csv

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from enum import IntEnum
from functools import reduce
from typing import Any, TextIO

from executorch.backends.test.harness.error_statistics import ErrorStatistics
from torch.export import ExportedProgram


# The maximum number of model output tensors to log statistics for. Most model tests will
# only have one output, but some may return more than one tensor. This upper bound is needed
# upfront since the file is written progressively. Any outputs beyond these will not have stats logged.
MAX_LOGGED_MODEL_OUTPUTS = 2


# Field names for the CSV report.
CSV_FIELD_NAMES = [
    "Test ID",
    "Test Case",
    "Subtest",
    "Flow",
    "Params",
    "Result",
    "Result Detail",
    "Delegated",
    "Quantize Time (s)",
    "Lower Time (s)",
    "Delegated Nodes",
    "Undelegated Nodes",
    "Delegated Ops",
    "Undelegated Ops",
    "PTE Size (Kb)",
]

for i in range(MAX_LOGGED_MODEL_OUTPUTS):
    CSV_FIELD_NAMES.extend(
        [
            f"Output {i} Error Max",
            f"Output {i} Error MAE",
            f"Output {i} SNR",
        ]
    )


# Operators that are excluded from the counts returned by count_ops. These are used to
# exclude operatations that are not logically relevant or delegatable to backends.
OP_COUNT_IGNORED_OPS = {
    "executorch_call_delegate",
    "getitem",
}


class TestResult(IntEnum):
    """Represents the result of a test case run, indicating success or a specific failure reason."""

    SUCCESS = 0
    """ The test succeeded with the backend delegate part or all of the graph. """

    SUCCESS_UNDELEGATED = 1
    """ The test succeeded without the backend delegating anything. """

    SKIPPED = 2
    """ The test was skipped due to a non-backend failure. """

    QUANTIZE_FAIL = 3
    """ The test failed due to the quantization stage failing. """

    LOWER_FAIL = 4
    """ The test failed due to a failure in partitioning or lowering. """

    PTE_LOAD_FAIL = 5
    """ The test failed due to the resulting PTE failing to load. """

    PTE_RUN_FAIL = 6
    """ The test failed due to the resulting PTE failing to run. """

    OUTPUT_MISMATCH_FAIL = 7
    """ The test failed due to a mismatch between runtime and reference outputs. """

    UNKNOWN_FAIL = 8
    """ The test failed in an unknown or unexpected manner. """

    def is_success(self):
        return self in {TestResult.SUCCESS, TestResult.SUCCESS_UNDELEGATED}

    def is_non_backend_failure(self):
        return self in {TestResult.SKIPPED}

    def is_backend_failure(self):
        return not self.is_success() and not self.is_non_backend_failure()

    def to_short_str(self):
        if self in {TestResult.SUCCESS, TestResult.SUCCESS_UNDELEGATED}:
            return "Pass"
        elif self == TestResult.SKIPPED:
            return "Skip"
        else:
            return "Fail"

    def to_detail_str(self):
        if self == TestResult.SUCCESS:
            return ""
        elif self == TestResult.SUCCESS_UNDELEGATED:
            return ""
        elif self == TestResult.SKIPPED:
            return ""
        elif self == TestResult.QUANTIZE_FAIL:
            return "Quantization Failed"
        elif self == TestResult.LOWER_FAIL:
            return "Lowering Failed"
        elif self == TestResult.PTE_LOAD_FAIL:
            return "PTE Load Failed"
        elif self == TestResult.PTE_RUN_FAIL:
            return "PTE Run Failed"
        elif self == TestResult.OUTPUT_MISMATCH_FAIL:
            return "Output Mismatch"
        elif self == TestResult.UNKNOWN_FAIL:
            return "Unknown Failure"
        else:
            raise ValueError(f"Invalid TestResult value: {self}.")

    def display_name(self):
        if self == TestResult.SUCCESS:
            return "Success (Delegated)"
        elif self == TestResult.SUCCESS_UNDELEGATED:
            return "Success (Undelegated)"
        elif self == TestResult.SKIPPED:
            return "Skipped"
        elif self == TestResult.QUANTIZE_FAIL:
            return "Fail (Quantize)"
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

    subtest_index: int
    """ The subtest number. If a test case runs multiple tests, this field can be used to disambiguate. """

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

    quantize_time: timedelta | None = None
    """ The total runtime of the quantization stage, or none, if the test did not run the quantize stage. """

    lower_time: timedelta | None = None
    """ The total runtime of the to_edge_transform_and_lower stage, or none, if the test did not run the quantize stage. """

    delegated_op_counts: Counter | None = None
    """ The number of delegated occurances of each operator in the graph. """

    undelegated_op_counts: Counter | None = None
    """ The number of undelegated occurances of each operator in the graph. """

    pte_size_bytes: int | None = None
    """ The size of the PTE file in bytes. """

    def is_delegated(self):
        return (
            any(v > 0 for v in self.delegated_op_counts.values())
            if self.delegated_op_counts
            else False
        )


@dataclass
class TestSessionState:
    seed: int

    # True if the CSV header has been written to report__path.
    has_written_report_header: bool = False

    # The file path to write the detail report to, if enabled.
    report_path: str | None = None

    test_case_summaries: list[TestCaseSummary] = field(default_factory=list)


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


def _get_target_name(target: Any) -> str:
    """Retrieve a string representation of a node target."""
    if isinstance(target, str):
        return target
    elif hasattr(target, "name"):
        return target.name()  # Op overloads have this
    elif hasattr(target, "__name__"):
        return target.__name__  # Some builtins have this
    else:
        return str(target)


def _count_ops(program: ExportedProgram) -> Counter:
    op_names = (
        _get_target_name(n.target)
        for n in program.graph.nodes
        if n.op == "call_function"
    )

    return Counter(op for op in op_names if op not in OP_COUNT_IGNORED_OPS)


def count_ops(program: dict[str, ExportedProgram] | ExportedProgram) -> Counter:
    if isinstance(program, ExportedProgram):
        return _count_ops(program)
    else:
        # Sum op counts for all methods in the program.
        return reduce(
            lambda a, b: a + b,
            (_count_ops(p) for p in program.values()),
            Counter(),
        )


def begin_test_session(report_path: str | None, seed: int):
    global _active_session

    assert _active_session is None, "A test session is already active."
    _active_session = TestSessionState(report_path=report_path, seed=seed)


def get_active_test_session() -> TestSessionState | None:
    global _active_session

    return _active_session


def log_test_summary(summary: TestCaseSummary):
    global _active_session

    if _active_session is not None:
        _active_session.test_case_summaries.append(summary)

        if _active_session.report_path is not None:
            file_mode = "a" if _active_session.has_written_report_header else "w"
            with open(_active_session.report_path, file_mode) as f:
                if not _active_session.has_written_report_header:
                    write_csv_header(f)
                    _active_session.has_written_report_header = True

                write_csv_row(summary, f)


def complete_test_session() -> RunSummary:
    global _active_session

    assert _active_session is not None, "No test session is active."
    summary = RunSummary.from_session(_active_session)
    _active_session = None

    return summary


def _sum_op_counts(counter: Counter | None) -> int | None:
    """
    A utility function to count the total number of nodes in an op count dict.
    """
    return sum(counter.values()) if counter is not None else None


def _serialize_params(params: dict[str, Any] | None) -> str:
    if params is not None:
        return str(dict(sorted(params.items())))
    else:
        return ""


def _serialize_op_counts(counter: Counter | None) -> str:
    """
    A utility function to serialize op counts to a string, for the purpose of including
    in the test report.
    """
    if counter is not None:
        return str(dict(sorted(counter.items())))
    else:
        return ""


def write_csv_header(output: TextIO):
    writer = csv.DictWriter(output, CSV_FIELD_NAMES)
    writer.writeheader()


def write_csv_row(record: TestCaseSummary, output: TextIO):
    writer = csv.DictWriter(output, CSV_FIELD_NAMES)

    row = {
        "Test ID": record.name,
        "Test Case": record.base_name,
        "Subtest": record.subtest_index,
        "Flow": record.flow,
        "Params": _serialize_params(record.params),
        "Result": record.result.to_short_str(),
        "Result Detail": record.result.to_detail_str(),
        "Delegated": "True" if record.is_delegated() else "False",
        "Quantize Time (s)": (
            f"{record.quantize_time.total_seconds():.3f}"
            if record.quantize_time
            else None
        ),
        "Lower Time (s)": (
            f"{record.lower_time.total_seconds():.3f}" if record.lower_time else None
        ),
    }

    for output_idx, error_stats in enumerate(record.tensor_error_statistics):
        if output_idx >= MAX_LOGGED_MODEL_OUTPUTS:
            print(
                f"Model output stats are truncated as model has more than {MAX_LOGGED_MODEL_OUTPUTS} outputs. Consider increasing MAX_LOGGED_MODEL_OUTPUTS."
            )
            break

        row[f"Output {output_idx} Error Max"] = f"{error_stats.error_max:.3f}"
        row[f"Output {output_idx} Error MAE"] = f"{error_stats.error_mae:.3f}"
        row[f"Output {output_idx} SNR"] = f"{error_stats.sqnr:.3f}"

    row["Delegated Nodes"] = _sum_op_counts(record.delegated_op_counts)
    row["Undelegated Nodes"] = _sum_op_counts(record.undelegated_op_counts)
    row["Delegated Ops"] = _serialize_op_counts(record.delegated_op_counts)
    row["Undelegated Ops"] = _serialize_op_counts(record.undelegated_op_counts)
    row["PTE Size (Kb)"] = (
        f"{record.pte_size_bytes / 1000.0:.3f}" if record.pte_size_bytes else ""
    )

    writer.writerow(row)
