#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run all MLX delegate op tests.

Usage:
    # Run all tests (all configurations):
    python -m executorch.backends.mlx.test.run_all_tests

    # Run specific test (all its configurations):
    python -m executorch.backends.mlx.test.run_all_tests add

    # Run specific test configuration:
    python -m executorch.backends.mlx.test.run_all_tests add_scalar

    # List available tests:
    python -m executorch.backends.mlx.test.run_all_tests --list

    # Rebuild C++ runner before running:
    python -m executorch.backends.mlx.test.run_all_tests --rebuild

    # Run tests in parallel:
    python -m executorch.backends.mlx.test.run_all_tests -j 4

    # Run with custom timeout:
    python -m executorch.backends.mlx.test.run_all_tests --timeout 60
"""

import argparse
import importlib
import multiprocessing
import subprocess
import sys
from multiprocessing import Pool
from typing import List, Optional, Tuple

from .test_utils import (
    clean_test_outputs,
    DEFAULT_TEST_TIMEOUT,
    get_all_test_configs,
    get_registered_tests,
    get_test_output_size,
    rebuild_op_test_runner,
)


def discover_and_import_tests():
    """
    Import test_ops.py module which contains all test definitions.
    This triggers registration of all tests.
    """
    importlib.import_module(".test_ops", package=__package__)


# =============================================================================
# Single Test Runner (for parallel execution)
# =============================================================================


def _run_single_test(
    test_class_name: str,
    config_name: str,
    config_kwargs: dict,
    verbose: bool,
    timeout: int,
) -> Tuple[str, bool, Optional[str]]:
    """
    Run a single test configuration in a subprocess.

    This function is designed to be called via ProcessPoolExecutor.
    It recreates the test instance from the class name and kwargs.

    Args:
        test_class_name: Name of the test class module.path
        config_name: Name of this configuration
        config_kwargs: Kwargs to recreate the test instance
        verbose: Whether to print verbose output
        timeout: Timeout in seconds

    Returns:
        (config_name, passed, error_message)
    """
    try:
        # Re-discover and import tests in this subprocess
        discover_and_import_tests()

        # Find the test config by name
        all_configs = get_all_test_configs()
        test_instance = None
        for name, instance in all_configs:
            if name == config_name:
                test_instance = instance
                break

        if test_instance is None:
            return (config_name, False, f"Could not find test config: {config_name}")

        # Run the test
        passed = test_instance.run_test(verbose=verbose, timeout=timeout)
        return (config_name, passed, None)

    except Exception as e:
        import traceback

        return (config_name, False, f"Exception: {e}\n{traceback.format_exc()}")


def run_tests_sequential(
    configs_to_run: List[Tuple[str, object]],
    verbose: bool = False,
    timeout: int = DEFAULT_TEST_TIMEOUT,
    clean_after_each: bool = False,
    isolate: bool = False,
) -> Tuple[int, int, List[str]]:
    """
    Run tests sequentially.

    Args:
        configs_to_run: List of (config_name, test_instance) tuples.
        verbose: Whether to print verbose output.
        timeout: Timeout in seconds per test.
        clean_after_each: Whether to clean up test outputs after each test.
        isolate: Whether to run each test in a subprocess to prevent memory
            accumulation across tests (torch/MLX/Metal allocations).

    Returns:
        (passed_count, failed_count, failed_test_names)
    """
    passed = 0
    failed = 0
    failed_tests = []

    for config_name, test in configs_to_run:
        if isolate:
            test_passed = _run_test_in_subprocess(
                config_name, verbose=verbose, timeout=timeout
            )
        else:
            try:
                test_passed = test.run_test(verbose=verbose, timeout=timeout)
            except Exception as e:
                print(f"✗ FAILED: {config_name} - Exception: {e}")
                import traceback

                traceback.print_exc()
                test_passed = False

        if test_passed:
            passed += 1
        else:
            failed += 1
            failed_tests.append(config_name)

        if clean_after_each:
            clean_test_outputs([config_name], verbose=False)

    return passed, failed, failed_tests


def _run_test_in_subprocess(
    config_name: str,
    verbose: bool = False,
    timeout: int = DEFAULT_TEST_TIMEOUT,
) -> bool:
    """
    Run a single test in an isolated subprocess.

    Each test gets its own Python interpreter so torch/MLX/Metal memory is
    fully released between tests, preventing OOM on CI runners.

    Args:
        config_name: Name of the test configuration to run.
        verbose: Whether to print verbose output.
        timeout: Timeout in seconds.

    Returns:
        True if test passed, False otherwise.
    """
    cmd = [
        sys.executable,
        "-m",
        "executorch.backends.mlx.test.test_utils",
        config_name,
        "run",
    ]
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"✗ FAILED: {config_name} - Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"✗ FAILED: {config_name} - Subprocess error: {e}")
        return False


def run_tests_parallel(
    configs_to_run: List[Tuple[str, object]],
    num_workers: int,
    verbose: bool = False,
    timeout: int = DEFAULT_TEST_TIMEOUT,
    max_tasks_per_worker: Optional[int] = None,
) -> Tuple[int, int, List[str]]:
    """
    Run tests in parallel using multiprocessing.Pool.

    Args:
        configs_to_run: List of (config_name, test_instance) tuples.
        num_workers: Number of parallel workers.
        verbose: Whether to print verbose output.
        timeout: Timeout in seconds per test.
        max_tasks_per_worker: Maximum tasks per worker before recycling.
            When set, worker processes are terminated and replaced after
            completing this many tests, which releases accumulated memory
            (torch/MLX/Metal allocations). None means workers are never recycled.

    Returns:
        (passed_count, failed_count, failed_test_names)
    """
    passed = 0
    failed = 0
    failed_tests = []

    # Prepare test args for parallel execution
    # We pass config names and let subprocesses recreate the test instances
    test_args = [("", name, {}, verbose, timeout) for name, _ in configs_to_run]

    recycle_msg = ""
    if max_tasks_per_worker is not None:
        recycle_msg = f", recycling workers every {max_tasks_per_worker} tests"
    print(
        f"\nRunning {len(test_args)} tests with {num_workers} workers{recycle_msg}...\n"
    )

    with Pool(processes=num_workers, maxtasksperchild=max_tasks_per_worker) as pool:
        results = pool.starmap(_run_single_test, test_args)

    for result_name, result_passed, error_msg in results:
        if result_passed:
            print(f"✓ PASSED: {result_name}")
            passed += 1
        else:
            if error_msg:
                print(f"✗ FAILED: {result_name} - {error_msg}")
            else:
                print(f"✗ FAILED: {result_name}")
            failed += 1
            failed_tests.append(result_name)

    return passed, failed, failed_tests


def run_tests(
    test_filter: List[str],
    verbose: bool = False,
    parallel: int = 1,
    timeout: int = DEFAULT_TEST_TIMEOUT,
    clean_after_each: bool = False,
    isolate: bool = False,
    max_tasks_per_worker: Optional[int] = None,
) -> Tuple[int, int, List[str]]:
    """
    Run tests matching the filter.

    Args:
        test_filter: List of test names/patterns to run. If empty, runs all tests.
            Can match either base test name (e.g., "add") or config name (e.g., "add_scalar").
        verbose: Whether to print verbose output.
        parallel: Number of parallel workers (1 = sequential).
        timeout: Timeout in seconds per test.
        clean_after_each: Whether to clean up test outputs after each test (sequential only).
        isolate: Whether to run each test in a subprocess (sequential only).
        max_tasks_per_worker: Maximum tasks per worker before recycling (parallel only).

    Returns:
        (passed_count, failed_count, failed_test_names)
    """
    all_configs = get_all_test_configs()
    registry = get_registered_tests()

    # Determine which configs to run
    configs_to_run = []
    if not test_filter:
        # Run all
        configs_to_run = all_configs
    else:
        for pattern in test_filter:
            matched = False

            # Check if pattern matches a base test name
            if pattern in registry:
                configs_to_run.extend(registry[pattern])
                matched = True
            else:
                # Check if pattern matches a config name
                for config_name, config in all_configs:
                    if config_name == pattern:
                        configs_to_run.append((config_name, config))
                        matched = True

            if not matched:
                print(f"Warning: No test matching '{pattern}', skipping")

    if not configs_to_run:
        print("No tests to run.")
        return 0, 0, []

    # Run tests
    if parallel > 1:
        return run_tests_parallel(
            configs_to_run, parallel, verbose, timeout, max_tasks_per_worker
        )
    else:
        return run_tests_sequential(
            configs_to_run, verbose, timeout, clean_after_each, isolate
        )


def main():  # noqa: C901
    # Get CPU count for default parallel workers
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(description="Run all MLX delegate op tests")
    parser.add_argument(
        "tests",
        nargs="*",
        help="Specific tests to run (default: all). Can be base name (e.g., 'add') or config name (e.g., 'add_scalar')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the C++ test runner before running",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up generated test files and exit",
    )
    parser.add_argument(
        "--clean-after",
        action="store_true",
        help="Clean up generated test files after running tests",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Run each test in a separate subprocess to prevent memory accumulation",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help=f"Run tests in parallel with N workers (default: 1, max: {cpu_count})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TEST_TIMEOUT,
        metavar="SECS",
        help=f"Timeout per test in seconds (default: {DEFAULT_TEST_TIMEOUT})",
    )
    parser.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=None,
        metavar="N",
        help="Recycle parallel workers after N tests to release memory (default: no recycling)",
    )
    args = parser.parse_args()

    # Validate parallel workers
    if args.parallel < 1:
        args.parallel = 1
    elif args.parallel > cpu_count:
        print(
            f"Warning: --parallel {args.parallel} exceeds CPU count ({cpu_count}), using {cpu_count}"
        )
        args.parallel = cpu_count

    # Auto-discover and import all test modules
    discover_and_import_tests()

    # Handle --clean flag
    if args.clean:
        # Determine which tests to clean
        test_names = None
        if args.tests:
            # Get config names for the specified tests
            registry = get_registered_tests()
            test_names = []
            for pattern in args.tests:
                if pattern in registry:
                    test_names.extend(cfg_name for cfg_name, _ in registry[pattern])
                else:
                    test_names.append(pattern)

        # Show current size
        current_size = get_test_output_size(test_names)
        if current_size > 0:
            print(f"Current test output size: {current_size / 1024 / 1024:.2f} MB")

        # Clean
        files_removed = clean_test_outputs(test_names, verbose=args.verbose)
        if files_removed > 0:
            print(f"Removed {files_removed} files")
        else:
            print("No files to clean")
        sys.exit(0)

    # List tests
    if args.list:
        registry = get_registered_tests()
        print("Available tests:")
        for base_name in sorted(registry.keys()):
            configs = registry[base_name]
            if len(configs) == 1 and configs[0][0] == base_name:
                # Single config with same name as base
                print(f"  {base_name}")
            else:
                # Multiple configs or different name
                print(f"  {base_name}:")
                for config_name, _ in configs:
                    print(f"    - {config_name}")
        sys.exit(0)

    # Rebuild if requested
    if args.rebuild:
        if not rebuild_op_test_runner(verbose=args.verbose):
            sys.exit(1)

    # Run tests
    passed, failed, failed_tests = run_tests(
        args.tests,
        verbose=args.verbose,
        parallel=args.parallel,
        timeout=args.timeout,
        clean_after_each=args.clean_after,
        isolate=args.isolate,
        max_tasks_per_worker=args.max_tasks_per_worker,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
    print("=" * 60)

    # Clean up after tests if requested
    if args.clean_after:
        # Determine which tests to clean (same logic as --clean)
        test_names = None
        if args.tests:
            registry = get_registered_tests()
            test_names = []
            for pattern in args.tests:
                if pattern in registry:
                    test_names.extend(cfg_name for cfg_name, _ in registry[pattern])
                else:
                    test_names.append(pattern)

        current_size = get_test_output_size(test_names)
        files_removed = clean_test_outputs(test_names, verbose=args.verbose)
        if files_removed > 0:
            print(
                f"\nCleaned up {files_removed} files ({current_size / 1024 / 1024:.2f} MB)"
            )

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
