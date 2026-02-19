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
    python -m executorch.backends.apple.mlx.test.run_all_tests

    # Run specific test (all its configurations):
    python -m executorch.backends.apple.mlx.test.run_all_tests add

    # Run specific test configuration:
    python -m executorch.backends.apple.mlx.test.run_all_tests add_scalar

    # List available tests:
    python -m executorch.backends.apple.mlx.test.run_all_tests --list

    # Rebuild C++ runner before running:
    python -m executorch.backends.apple.mlx.test.run_all_tests --rebuild

    # Run tests in parallel:
    python -m executorch.backends.apple.mlx.test.run_all_tests -j 4

    # Run with custom timeout:
    python -m executorch.backends.apple.mlx.test.run_all_tests --timeout 60
"""

import argparse
import importlib
import multiprocessing
import os
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
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
) -> Tuple[int, int, List[str]]:
    """
    Run tests sequentially.

    Args:
        configs_to_run: List of (config_name, test_instance) tuples.
        verbose: Whether to print verbose output.
        timeout: Timeout in seconds per test.

    Returns:
        (passed_count, failed_count, failed_test_names)
    """
    passed = 0
    failed = 0
    failed_tests = []

    for config_name, test in configs_to_run:
        try:
            if test.run_test(verbose=verbose, timeout=timeout):
                passed += 1
            else:
                failed += 1
                failed_tests.append(config_name)
        except Exception as e:
            print(f"✗ FAILED: {config_name} - Exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
            failed_tests.append(config_name)

    return passed, failed, failed_tests


def run_tests_parallel(
    configs_to_run: List[Tuple[str, object]],
    num_workers: int,
    verbose: bool = False,
    timeout: int = DEFAULT_TEST_TIMEOUT,
) -> Tuple[int, int, List[str]]:
    """
    Run tests in parallel using ProcessPoolExecutor.

    Args:
        configs_to_run: List of (config_name, test_instance) tuples.
        num_workers: Number of parallel workers.
        verbose: Whether to print verbose output.
        timeout: Timeout in seconds per test.

    Returns:
        (passed_count, failed_count, failed_test_names)
    """
    passed = 0
    failed = 0
    failed_tests = []

    # Prepare test configs for parallel execution
    # We pass config names and let subprocesses recreate the test instances
    test_configs = [(name, {}) for name, _ in configs_to_run]

    print(f"\nRunning {len(test_configs)} tests with {num_workers} workers...\n")

    # Use explicit shutdown instead of context manager to avoid crashes during
    # worker teardown. Workers import torch/MLX/Metal and can segfault during
    # Python cleanup (atexit, __del__). With the context manager, that crash
    # raises BrokenProcessPool from __exit__, killing the main process before
    # it can print the test summary. Using wait=False lets us collect all
    # results and move on without blocking on worker process teardown.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    try:
        futures = {}
        for config_name, config_kwargs in test_configs:
            future = executor.submit(
                _run_single_test,
                "",  # test_class_name (not used, we find by config_name)
                config_name,
                config_kwargs,
                verbose,
                timeout,
            )
            futures[future] = config_name

        for future in as_completed(futures):
            config_name = futures[future]
            try:
                result_name, result_passed, error_msg = future.result()
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
            except Exception as e:
                print(f"✗ FAILED: {config_name} - Worker exception: {e}")
                failed += 1
                failed_tests.append(config_name)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return passed, failed, failed_tests


def run_tests(
    test_filter: List[str],
    verbose: bool = False,
    parallel: int = 1,
    timeout: int = DEFAULT_TEST_TIMEOUT,
) -> Tuple[int, int, List[str]]:
    """
    Run tests matching the filter.

    Args:
        test_filter: List of test names/patterns to run. If empty, runs all tests.
            Can match either base test name (e.g., "add") or config name (e.g., "add_scalar").
        verbose: Whether to print verbose output.
        parallel: Number of parallel workers (1 = sequential).
        timeout: Timeout in seconds per test.

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
        return run_tests_parallel(configs_to_run, parallel, verbose, timeout)
    else:
        return run_tests_sequential(configs_to_run, verbose, timeout)


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

    # Flush and use os._exit() to avoid ProcessPoolExecutor's atexit handler
    # which joins worker threads. Workers that imported torch/MLX/Metal can
    # segfault during Python cleanup, causing the atexit join to hang or crash.
    # In CI (GitHub Actions), stdout is pipe-buffered so we must flush explicitly
    # before os._exit() which does not flush stdio buffers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
