import argparse
import unittest

from typing import Callable

import torch

from executorch.backends.test.harness import Tester
from executorch.backends.test.suite.reporting import (
    begin_test_session,
    complete_test_session,
    RunSummary,
    TestCaseSummary,
    TestResult,
)


def run_test(  # noqa: C901
    model: torch.nn.Module,
    inputs: any,
    tester_factory: Callable[[], Tester],
    test_name: str,
    flow_name: str,
    params: dict | None,
) -> TestCaseSummary:
    """
    Top-level test run function for a model, input set, and tester. Handles test execution
    and reporting.
    """

    # Helper method to construct the summary.
    def build_result(
        result: TestResult, error: Exception | None = None
    ) -> TestCaseSummary:
        return TestCaseSummary(
            name=test_name,
            flow=flow_name,
            params=params,
            result=result,
            error=error,
        )

    # Ensure the model can run in eager mode.
    try:
        model(*inputs)
    except Exception as e:
        return build_result(TestResult.EAGER_FAIL, e)

    try:
        tester = tester_factory(model, inputs)
    except Exception as e:
        return build_result(TestResult.UNKNOWN_FAIL, e)

    try:
        tester.export()
    except Exception as e:
        return build_result(TestResult.EXPORT_FAIL, e)

    try:
        tester.to_edge_transform_and_lower()
    except Exception as e:
        return build_result(TestResult.LOWER_FAIL, e)

    is_delegated = any(
        n.target == torch._higher_order_ops.executorch_call_delegate
        for n in tester.stages[tester.cur].graph_module.graph.nodes
        if n.op == "call_function"
    )

    # Only run the runtime portion if something was delegated.
    if is_delegated:
        try:
            tester.to_executorch().serialize()
        except Exception as e:
            # We could introduce a result value for this, but I'm not sure it's necessary.
            # We can do this if we ever see to_executorch() or serialize() fail due a backend issue.
            return build_result(TestResult.UNKNOWN_FAIL, e)

        # TODO We should consider refactoring the tester slightly to return more signal on
        # the cause of a failure in run_method_and_compare_outputs. We can look for
        # AssertionErrors to catch output mismatches, but this might catch more than that.
        try:
            tester.run_method_and_compare_outputs()
        except AssertionError as e:
            return build_result(TestResult.OUTPUT_MISMATCH_FAIL, e)
        except Exception as e:
            return build_result(TestResult.PTE_RUN_FAIL, e)
    else:
        return build_result(TestResult.SUCCESS_UNDELEGATED)

    return build_result(TestResult.SUCCESS)


def print_summary(summary: RunSummary):
    print()
    print("Test Session Summary:")

    print()
    print(f"{summary.total_passed:>5} Passed / {summary.num_test_cases}")
    print(f"{summary.total_failed:>5} Failed / {summary.num_test_cases}")
    print(f"{summary.total_skipped:>5} Skipped / {summary.num_test_cases}")

    print()
    print("[Success]")
    print(f"{summary.aggregated_results.get(TestResult.SUCCESS, 0):>5} Delegated")
    print(
        f"{summary.aggregated_results.get(TestResult.SUCCESS_UNDELEGATED, 0):>5} Undelegated"
    )

    print()
    print("[Failure]")
    print(
        f"{summary.aggregated_results.get(TestResult.LOWER_FAIL, 0):>5} Lowering Fail"
    )
    print(
        f"{summary.aggregated_results.get(TestResult.PTE_LOAD_FAIL, 0):>5} PTE Load Fail"
    )
    print(
        f"{summary.aggregated_results.get(TestResult.PTE_RUN_FAIL, 0):>5} PTE Run Fail"
    )
    print(
        f"{summary.aggregated_results.get(TestResult.OUTPUT_MISMATCH_FAIL, 0):>5} Output Mismatch Fail"
    )

    print()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ExecuTorch Backend Test Suite",
        description="Run ExecuTorch backend tests.",
    )
    parser.add_argument("test_path", nargs="?", help="Prefix filter for tests to run.")
    return parser.parse_args()


def runner_main():
    args = parse_args()

    begin_test_session()

    test_path = args.test_path or "executorch.backends.test.suite.operators"

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_path)
    unittest.TextTestRunner().run(suite)

    summary = complete_test_session()
    print_summary(summary)


if __name__ == "__main__":
    runner_main()
