# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import sys
from pathlib import Path

from executorch.backends.arm.scripts import evaluate_model


def _run_evaluate_model(*args: str) -> None:
    previous_argv = sys.argv
    try:
        sys.argv = ["evaluate_model.py", *args]
        evaluate_model.main()
    finally:
        sys.argv = previous_argv


def test_evaluate_model_tosa_INT(tmp_path: Path) -> None:
    intermediates = tmp_path / "test_evaluate_model_tosa_INT_intermediates"
    output = tmp_path / "test_evaluate_model_tosa_INT_metrics.json"

    _run_evaluate_model(
        "--model_name",
        "add",
        "--target",
        "TOSA-1.0+INT",
        "--quant_mode",
        "int8",
        "--no_delegate",
        "--evaluators",
        "numerical",
        "--intermediates",
        str(intermediates),
        "--output",
        str(output),
    )

    assert output.exists(), f"Metrics file not created at {output}"
    data = json.loads(output.read_text())
    assert data["name"] == "add"
    assert "metrics" in data
    assert "mean_absolute_error" in data["metrics"]


def test_evaluate_model_tosa_FP(tmp_path: Path) -> None:
    intermediates = tmp_path / "test_evaluate_model_tosa_FP_intermediates"
    output = tmp_path / "test_evaluate_model_tosa_FP_metrics.json"

    _run_evaluate_model(
        "--model_name",
        "add",
        "--target",
        "TOSA-1.0+FP",
        "--evaluators",
        "numerical",
        "--intermediates",
        str(intermediates),
        "--output",
        str(output),
    )

    assert output.exists(), f"Metrics file not created at {output}"
    data = json.loads(output.read_text())
    assert data["name"] == "add"
    assert "metrics" in data
    assert "mean_absolute_error" in data["metrics"]
    assert "compression_ratio" in data["metrics"]
