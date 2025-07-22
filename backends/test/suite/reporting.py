from collections import Counter
from dataclasses import dataclass
from enum import IntEnum


class TestResult(IntEnum):
    """Represents the result of a test case run, indicating success or a specific failure reason."""

    SUCCESS = 0
    """ The test succeeded with the backend delegate part or all of the graph. """

    SUCCESS_UNDELEGATED = 1
    """ The test succeeded without the backend delegating anything. """

    EAGER_FAIL = 2
    """ The test failed due to the model failing to run in eager mode. """

    EXPORT_FAIL = 3
    """ The test failed due to the model failing to export. """

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

    name: str
    """ The qualified name of the test, not including the flow suffix. """

    flow: str
    """ The backend-specific flow name. Corresponds to flows registered in backends/test/suite/__init__.py. """

    params: dict | None
    """ Test-specific parameters, such as dtype. """

    result: TestResult
    """ The top-level result, such as SUCCESS or LOWER_FAIL. """

    error: Exception | None
    """ The Python exception object, if any. """


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
