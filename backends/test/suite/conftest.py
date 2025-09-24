from typing import Any

import pytest
import torch

from executorch.backends.test.suite.flow import all_flows
from executorch.backends.test.suite.reporting import _sum_op_counts
from executorch.backends.test.suite.runner import run_test


def pytest_configure(config):
    backends = set()

    for flow in all_flows().values():
        config.addinivalue_line(
            "markers",
            f"flow_{flow.name}: mark a test as testing the {flow.name} flow",
        )

        if flow.backend not in backends:
            config.addinivalue_line(
                "markers",
                f"backend_{flow.backend}: mark a test as testing the {flow.backend} backend",
            )
            backends.add(flow.backend)


class TestRunner:
    def __init__(self, flow, test_name, test_base_name):
        self._flow = flow
        self._test_name = test_name
        self._test_base_name = test_base_name
        self._subtest = 0
        self._results = []

    def lower_and_run_model(
        self,
        model: torch.nn.Module,
        inputs: Any,
        generate_random_test_inputs=True,
        dynamic_shapes=None,
    ):
        run_summary = run_test(
            model,
            inputs,
            self._flow,
            self._test_name,
            self._test_base_name,
            self._subtest,
            None,
            generate_random_test_inputs=generate_random_test_inputs,
            dynamic_shapes=dynamic_shapes,
        )

        self._subtest += 1
        self._results.append(run_summary)

        if not run_summary.result.is_success():
            if run_summary.result.is_backend_failure():
                raise RuntimeError("Test failure.") from run_summary.error
            else:
                # Non-backend failure indicates a bad test. Mark as skipped.
                pytest.skip(
                    f"Test failed for reasons other than backend failure. Error: {run_summary.error}"
                )


@pytest.fixture(
    params=[
        pytest.param(
            f,
            marks=[
                getattr(pytest.mark, f"flow_{f.name}"),
                getattr(pytest.mark, f"backend_{f.backend}"),
            ],
        )
        for f in all_flows().values()
    ],
    ids=str,
)
def test_runner(request):
    return TestRunner(request.param, request.node.name, request.node.originalname)


@pytest.hookimpl(optionalhook=True)
def pytest_json_runtest_metadata(item, call):
    metadata = {"subtests": []}

    if hasattr(item, "funcargs") and "test_runner" in item.funcargs:
        runner_instance = item.funcargs["test_runner"]

        for record in runner_instance._results:
            subtest_metadata = {}

            error_message = ""
            if record.error is not None:
                error_str = str(record.error)
                if len(error_str) > 400:
                    error_message = error_str[:200] + "..." + error_str[-200:]
                else:
                    error_message = error_str

            subtest_metadata["Test ID"] = record.name
            subtest_metadata["Test Case"] = record.base_name
            subtest_metadata["Subtest"] = record.subtest_index
            subtest_metadata["Flow"] = record.flow
            subtest_metadata["Result"] = record.result.to_short_str()
            subtest_metadata["Result Detail"] = record.result.to_detail_str()
            subtest_metadata["Error"] = error_message
            subtest_metadata["Delegated"] = "True" if record.is_delegated() else "False"
            subtest_metadata["Quantize Time (s)"] = (
                f"{record.quantize_time.total_seconds():.3f}"
                if record.quantize_time
                else None
            )
            subtest_metadata["Lower Time (s)"] = (
                f"{record.lower_time.total_seconds():.3f}"
                if record.lower_time
                else None
            )

            for output_idx, error_stats in enumerate(record.tensor_error_statistics):
                subtest_metadata[f"Output {output_idx} Error Max"] = (
                    f"{error_stats.error_max:.3f}"
                )
                subtest_metadata[f"Output {output_idx} Error MAE"] = (
                    f"{error_stats.error_mae:.3f}"
                )
                subtest_metadata[f"Output {output_idx} SNR"] = f"{error_stats.sqnr:.3f}"

            subtest_metadata["Delegated Nodes"] = _sum_op_counts(
                record.delegated_op_counts
            )
            subtest_metadata["Undelegated Nodes"] = _sum_op_counts(
                record.undelegated_op_counts
            )
            if record.delegated_op_counts:
                subtest_metadata["Delegated Ops"] = dict(record.delegated_op_counts)
            if record.undelegated_op_counts:
                subtest_metadata["Undelegated Ops"] = dict(record.undelegated_op_counts)
            subtest_metadata["PTE Size (Kb)"] = (
                f"{record.pte_size_bytes / 1000.0:.3f}" if record.pte_size_bytes else ""
            )

            metadata["subtests"].append(subtest_metadata)

    return metadata
