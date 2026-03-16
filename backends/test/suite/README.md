# Backend Test Suite

This directory contains tests that validate correctness and coverage of backends. These tests are written such that the backend is treated as a black box. The test logic verifies that the backend is able to handle a given pattern without erroring out (not partitioning is fine) and is able to run the graphs and yield reasonable outputs. As backends may differ significantly in implementation, numerical bounds are intentionally left loose.

These tests are intended to ensure that backends are robust and provide a smooth, "out-of-box" experience for users across the full span of input patterns. They are not intended to be a replacement for backend-specific tests, as they do not attempt to validate performance or that backends delegate operators that they expect to.

## Running Tests and Interpreting Output
Tests can be run from the command line using pytest. When generating a JSON test report, the runner will report detailed test statistics, including output accuracy, delegated nodes, lowering timing, and more.

Each backend and test flow (recipe) registers a pytest [marker](https://docs.pytest.org/en/stable/example/markers.html) that can be passed to pytest with the `-m marker` argument to filter execution.

To run all XNNPACK backend operator tests:
```
pytest -c /dev/nul backends/test/suite/operators/ -m backend_xnnpack -n auto
```

To run all model tests for the CoreML static int8 lowering flow:
```
pytest -c /dev/nul backends/test/suite/models/ -m flow_coreml_static_int8 -n auto
```

To run a specific test:
```
pytest -c /dev/nul backends/test/suite/ -k "test_prelu_f32_custom_init[xnnpack]"
```

To generate a JSON report:
```
pytest -c /dev/nul backends/test/suite/operators/ -n auto --json-report --json-report-file="test_report.json"
```

See [pytest-json-report](https://pypi.org/project/pytest-json-report/) for information on the report format. The test logic in this repository attaches additional metadata to each test entry under the `metadata`/`subtests` keys. One entry is created for each call to `test_runner.lower_and_run_model`.

Here is a excerpt from a test run, showing a successful run of the `test_add_f32_bcast_first[xnnpack]` test.
```json
"tests": [
    {
      "nodeid": "operators/test_add.py::test_add_f32_bcast_first[xnnpack]",
      "lineno": 38,
      "outcome": "passed",
      "keywords": [
        "test_add_f32_bcast_first[xnnpack]",
        "flow_xnnpack",
        "backend_xnnpack",
        ...
      ],
      "metadata": {
        "subtests": [
          {
            "Test ID": "test_add_f32_bcast_first[xnnpack]",
            "Test Case": "test_add_f32_bcast_first",
            "Subtest": 0,
            "Flow": "xnnpack",
            "Result": "Pass",
            "Result Detail": "",
            "Error": "",
            "Delegated": "True",
            "Quantize Time (s)": null,
            "Lower Time (s)": "2.881",
            "Output 0 Error Max": "0.000",
            "Output 0 Error MAE": "0.000",
            "Output 0 SNR": "inf",
            "Delegated Nodes": 1,
            "Undelegated Nodes": 0,
            "Delegated Ops": {
              "aten::add.Tensor": 1
            },
            "PTE Size (Kb)": "1.600"
          }
        ]
      }
```

## Backend Registration

To plug into the test framework, each backend should provide an implementation of the Tester class, defined in backends/test/harness/tester.py. Backends can provide implementations of each stage, or use the default implementation, as appropriate.

At a minimum, the backend will likely need to provide a custom implementation of the Partition and ToEdgeTransformAndLower stages using the appropriate backend partitioner. See backends/xnnpack/test/tester/tester.py for an example implementation.

Once a tester is available, the backend flow(s) can be added under flows/ and registered in flow.py. It is intended that this will be unified with the lowering recipes under executorch/export in the near future.

## Test Cases

Operator test cases are defined under the operators/ directory. Model tests are under models/. Tests are written in a backend-independent manner, and each test is programmatically expanded to generate a variant for each registered backend flow by use of the `test_runner` fixture parameter. Tests can additionally be parameterized using standard pytest decorators. Parameterizing over dtype is a common use case.

## Evolution of this Test Suite

This test suite is experimental and under active development. Tests are subject to added, removed, or modified without notice. It is anticipated that this suite will be stabilized by the 1.0 release of ExecuTorch.

There is currently no expectation that all backends pass all tests, as the content of the test suite is under development and open questions remain on error reporting, accuracy thresholds, and more.
