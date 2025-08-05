# Backend Test Suite

This directory contains tests that validate correctness and coverage of backends. These tests are written such that the backend is treated as a black box. The test logic verifies that the backend is able to handle a given pattern without erroring out (not partitioning is fine) and is able to run the graphs and yield reasonable outputs. As backends may differ significantly in implementation, numerical bounds are intentionally left loose.

These tests are intended to ensure that backends are robust and provide a smooth, "out-of-box" experience for users across the full span of input patterns. They are not intended to be a replacement for backend-specific tests, as they do not attempt to validate performance or that backends delegate operators that they expect to.

## Running Tests and Interpreting Output
Tests can be run from the command line, either using the runner.py entry point or the standard Python unittest runner. When running through runner.py, the test runner will report test statistics, including the number of tests with each result type.

Backends can be specified with the `ET_TEST_ENABLED_BACKENDS` environment variable. By default, all available backends are enabled. Note that backends such as Core ML or Vulkan may require specific hardware or software to be available. See the documentation for each backend for information on requirements.

Example:
```
ET_TEST_ENABLED_BACKENDS=xnnpack python -m executorch.backends.test.suite.runner
```

```
2465 Passed / 2494
16 Failed
13 Skipped

[Success]
736 Delegated
1729 Undelegated

[Failure]
5 Lowering Fail
3 PTE Run Fail
8 Output Mismatch Fail
```

Outcomes can be interpreted as follows:
 * Success (delegated): The test passed and at least one op was delegated by the backend.
 * Success (undelegated): The test passed with no ops delegated by the backend. This is a pass, as the partitioner works as intended.
 * Skipped: test fails in eager or export (indicative of a test or dynamo issue).
 * Lowering fail: The test fails in to_edge_transform_and_lower.
 * PTE run failure: The test errors out when loading or running the method.
 * Output mismatch failure: Output delta (vs eager) exceeds the configured tolerance.

## Backend Registration

To plug into the test framework, each backend should provide an implementation of the Tester class, defined in backends/test/harness/tester.py. Backends can provide implementations of each stage, or use the default implementation, as appropriate.

At a minimum, the backend will likely need to provide a custom implementation of the Partition and ToEdgeTransformAndLower stages using the appropriate backend partitioner. See backends/xnnpack/test/tester/tester.py for an example implementation.

Once a tester is available, the backend flow(s) can be added in __init__.py in this directory by adding an entry to `ALL_TESTER_FLOWS`. Each flow entry consists of a name (used in the test case naming) and a function to instantiate a tester for a given model and input tuple.

## Test Cases

Operator test cases are defined under the operators/ directory. Tests are written in a backend-independent manner, and each test is programmatically expanded to generate a variant for each registered backend flow. The `@operator_test` decorator is applied to each test class to trigger this behavior. Tests can also be tagged with an appropriate type specifier, such as `@dtype_test`, to generate variants for each dtype. The decorators and "magic" live in __init__.py in this directory.

## Evolution of this Test Suite

This test suite is experimental and under active development. Tests are subject to added, removed, or modified without notice. It is anticipated that this suite will be stabilized by the 1.0 release of ExecuTorch.

There is currently no expectation that all backends pass all tests, as the content of the test suite is under development and open questions remain on error reporting, accuracy thresholds, and more.
