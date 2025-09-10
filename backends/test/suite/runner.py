import argparse
import hashlib
import importlib
import random
import re
import time
import unittest
import warnings

from datetime import timedelta
from typing import Any

import torch

# Set of unsupported ops that should cause tests to be skipped
UNSUPPORTED_PORTABLE_OPS = {
    "aten::_embedding_bag",
    "aten::_adaptive_avg_pool2d",
    "aten::median",
    "aten::median.dim",
    "aten::round.decimals",
}

from executorch.backends.test.harness.error_statistics import ErrorStatistics
from executorch.backends.test.harness.stages import StageType
from executorch.backends.test.suite.discovery import discover_tests, TestFilter
from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.test.suite.reporting import (
    begin_test_session,
    complete_test_session,
    count_ops,
    get_active_test_session,
    RunSummary,
    TestCaseSummary,
    TestResult,
)
from executorch.exir import EdgeProgramManager
from executorch.exir.dialects._ops import ops as exir_ops


# A list of all runnable test suites and the corresponding python package.
NAMED_SUITES = {
    "models": "executorch.backends.test.suite.models",
    "operators": "executorch.backends.test.suite.operators",
}


def _graph_has_unsupported_patterns(program: torch.export.ExportedProgram) -> bool:
    # Returns true if the model contains patterns that will fail when running on the ET
    # portable kernel library.

    # Check for 3d convolutions. All convs (1d, 2d, 3d) use the same op, so we need to look at
    # the input meta to determine the rank.
    for node in program.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == exir_ops.edge.aten.convolution.default
        ):
            in_rank = node.args[0].meta["val"].dim()
            if in_rank != 4:
                return True

    return False


def _get_test_seed(test_base_name: str) -> int:
    # Set the seed based on the test base name to give consistent inputs between backends. Add the
    # run seed to allow for reproducible results, but still allow for run-to-run variation.
    # Having a stable hash between runs and across machines is a plus (builtin python hash is not).
    # Using MD5 here because it's fast and we don't actually care about cryptographic properties.
    test_session = get_active_test_session()
    run_seed = (
        test_session.seed
        if test_session is not None
        else random.randint(0, 100_000_000)
    )

    hasher = hashlib.md5()
    data = test_base_name.encode("utf-8")
    hasher.update(data)
    # Torch doesn't like very long seeds.
    return (int.from_bytes(hasher.digest(), "little") % 100_000_000) + run_seed


def run_test(  # noqa: C901
    model: torch.nn.Module,
    inputs: Any,
    flow: TestFlow,
    test_name: str,
    test_base_name: str,
    subtest_index: int,
    params: dict | None,
    dynamic_shapes: Any | None = None,
    generate_random_test_inputs: bool = True,
) -> TestCaseSummary:
    """
    Top-level test run function for a model, input set, and tester. Handles test execution
    and reporting.
    """

    error_statistics: list[ErrorStatistics] = []
    extra_stats = {}

    torch.manual_seed(_get_test_seed(test_base_name))

    # Helper method to construct the summary.
    def build_result(
        result: TestResult, error: Exception | None = None
    ) -> TestCaseSummary:
        return TestCaseSummary(
            backend=flow.backend,
            base_name=test_base_name,
            subtest_index=subtest_index,
            flow=flow.name,
            name=test_name,
            params=params,
            result=result,
            error=error,
            tensor_error_statistics=error_statistics,
            **extra_stats,
        )

    # Ensure the model can run in eager mode.
    try:
        model(*inputs)
    except Exception as e:
        return build_result(TestResult.SKIPPED, e)

    try:
        tester = flow.tester_factory(model, inputs)
    except Exception as e:
        return build_result(TestResult.UNKNOWN_FAIL, e)

    if flow.quantize:
        start_time = time.perf_counter()
        try:
            tester.quantize(
                flow.quantize_stage_factory() if flow.quantize_stage_factory else None
            )
            elapsed = time.perf_counter() - start_time
            extra_stats["quantize_time"] = timedelta(seconds=elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            extra_stats["quantize_time"] = timedelta(seconds=elapsed)
            return build_result(TestResult.QUANTIZE_FAIL, e)

    try:
        # TODO Use Tester dynamic_shapes parameter once input generation can properly handle derived dims.
        tester.export(
            tester._get_default_stage(StageType.EXPORT, dynamic_shapes=dynamic_shapes),
        )
    except Exception as e:
        return build_result(TestResult.SKIPPED, e)

    lower_start_time = time.perf_counter()
    try:
        tester.to_edge_transform_and_lower(generate_etrecord=True)
        elapsed = time.perf_counter() - lower_start_time
        extra_stats["lower_time"] = timedelta(seconds=elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - lower_start_time
        extra_stats["lower_time"] = timedelta(seconds=elapsed)
        return build_result(TestResult.LOWER_FAIL, e)

    # Compute delegation statistics. Use the ETRecord to access the edge dialect graph between
    # to_edge and delegation. Note that ETRecord only stores the edge dialect graph for a single
    # method currently and assumes it is called "forward".
    edge_manager: EdgeProgramManager = tester.get_artifact()
    edge_op_counts = count_ops({"forward": edge_manager._etrecord.edge_dialect_program})
    undelegated_op_counts = count_ops(edge_manager._edge_programs)
    delegated_op_counts = edge_op_counts - undelegated_op_counts

    extra_stats["delegated_op_counts"] = delegated_op_counts
    extra_stats["undelegated_op_counts"] = undelegated_op_counts

    is_delegated = any(
        n.target == torch._higher_order_ops.executorch_call_delegate
        for n in tester.stages[tester.cur].graph_module.graph.nodes
        if n.op == "call_function"
    )

    # Check if any undelegated ops are in the unsupported ops set.
    has_unsupported_ops = any(
        op in UNSUPPORTED_PORTABLE_OPS for op in undelegated_op_counts.keys()
    ) or _graph_has_unsupported_patterns(edge_manager._etrecord.edge_dialect_program)

    # Skip the test if there are unsupported portable ops remaining.
    if has_unsupported_ops:
        return build_result(TestResult.SKIPPED)

    # Only run the runtime portion if something was delegated (or the flow doesn't delegate)
    if is_delegated or not flow.is_delegated:
        try:
            tester.to_executorch().serialize()
            extra_stats["pte_size_bytes"] = len(tester.get_artifact())
        except Exception as e:
            # We could introduce a result value for this, but I'm not sure it's necessary.
            # We can do this if we ever see to_executorch() or serialize() fail due a backend issue.
            return build_result(TestResult.UNKNOWN_FAIL, e)

        # TODO We should consider refactoring the tester slightly to return more signal on
        # the cause of a failure in run_method_and_compare_outputs. We can look for
        # AssertionErrors to catch output mismatches, but this might catch more than that.
        try:
            tester.run_method_and_compare_outputs(
                inputs=None if generate_random_test_inputs else inputs,
                statistics_callback=lambda stats: error_statistics.append(stats),
                atol=1e-1,
                rtol=4e-2,
            )
        except AssertionError as e:
            return build_result(TestResult.OUTPUT_MISMATCH_FAIL, e)
        except Exception as e:
            return build_result(TestResult.PTE_RUN_FAIL, e)
    else:
        # Skip the test if nothing is delegated
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
        f"{summary.aggregated_results.get(TestResult.QUANTIZE_FAIL, 0):>5} Quantization Fail"
    )
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
    parser.add_argument(
        "suite",
        nargs="*",
        help="The test suite to run.",
        choices=NAMED_SUITES.keys(),
        default=["operators"],
    )
    parser.add_argument(
        "-b", "--backend", nargs="*", help="The backend or backends to test."
    )
    parser.add_argument("-l", "--flow", nargs="*", help="The flow or flows to test.")
    parser.add_argument(
        "-f", "--filter", nargs="?", help="A regular expression filter for test names."
    )
    parser.add_argument(
        "-r",
        "--report",
        nargs="?",
        help="A file to write the test report to, in CSV format.",
        default="backend_test_report.csv",
    )
    parser.add_argument(
        "--seed",
        nargs="?",
        help="The numeric seed value to use for random generation.",
        type=int,
    )
    return parser.parse_args()


def build_test_filter(args: argparse.Namespace) -> TestFilter:
    return TestFilter(
        backends=set(args.backend) if args.backend is not None else None,
        flows=set(args.flow) if args.flow is not None else None,
        name_regex=re.compile(args.filter) if args.filter is not None else None,
    )


def runner_main():
    args = parse_args()

    # Suppress deprecation warnings for export_for_training, as it generates a
    # lot of log spam. We don't really need the warning here.
    warnings.simplefilter("ignore", category=FutureWarning)

    seed = args.seed or random.randint(0, 100_000_000)
    print(f"Running with seed {seed}.")

    begin_test_session(args.report, seed=seed)

    if len(args.suite) > 1:
        raise NotImplementedError("TODO Support multiple suites.")

    test_path = NAMED_SUITES[args.suite[0]]
    test_root = importlib.import_module(test_path)
    test_filter = build_test_filter(args)

    suite = discover_tests(test_root, test_filter)
    unittest.TextTestRunner(verbosity=2).run(suite)

    summary = complete_test_session()
    print_summary(summary)


if __name__ == "__main__":
    runner_main()
