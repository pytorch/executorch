# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "cuda.yml").read_text())


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _model_quant(entry):
    return (entry["model"]["repo"], entry["model"]["name"], entry["quant"])


class CudaWorkflowTest(unittest.TestCase):
    def test_pybind_runs_inline_for_the_expected_matrix_cells(self):
        job = WORKFLOW["jobs"]["test-model-cuda-e2e"]
        matrix = job["strategy"]["matrix"]
        pybind_rows = [row for row in matrix["include"] if "pybind_model" in row]

        actual = {
            (*_model_quant(row), row["pybind_model"], row["pybind_quantized"])
            for row in pybind_rows
        }
        expected = {
            (
                "google",
                "gemma-3-4b-it",
                "quantized-int4-tile-packed",
                "gemma3-4b",
                True,
            ),
            (
                "Qwen",
                "Qwen3-0.6B",
                "non-quantized",
                "qwen3-0.6b",
                False,
            ),
            (
                "Qwen",
                "Qwen3-0.6B",
                "quantized-int4-tile-packed",
                "qwen3-0.6b",
                True,
            ),
        }
        self.assertEqual(expected, actual)

        excluded = {_model_quant(row) for row in matrix["exclude"]}
        active = {
            (model["repo"], model["name"], quant)
            for model in matrix["model"]
            for quant in matrix["quant"]
        } - excluded
        self.assertTrue({_model_quant(row) for row in pybind_rows} <= active)

        script = job["with"]["script"]
        self.assertIn('if [ -n "${{ matrix.pybind_model }}" ]', script)
        self.assertIn("test_huggingface_optimum_model.py", script)
        self.assertIn("--run_only", script)
        self.assertGreaterEqual(script.count('"${MODEL_DIR}"'), 2)

    def test_model_e2e_does_not_transfer_artifacts(self):
        self.assertNotIn("test-cuda-pybind", WORKFLOW["jobs"])
        keys = set(_all_keys(WORKFLOW["jobs"]["test-model-cuda-e2e"]))
        self.assertNotIn("upload-artifact", keys)
        self.assertNotIn("download-artifact", keys)
