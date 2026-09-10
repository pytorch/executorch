# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
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
    def test_build_matrix_preserves_existing_cuda_versions(self):
        job = WORKFLOW["jobs"]["test-cuda-builds"]
        self.assertEqual(
            job["strategy"]["matrix"]["cuda-version"], ["12.6", "13.0", "13.4"]
        )
        self.assertEqual(job["with"]["gpu-arch-version"], "${{ matrix.cuda-version }}")

    def test_cuda134_driver_uses_matching_workflow_and_action_revision(self):
        job = WORKFLOW["jobs"]["test-cuda-builds"]
        workflow, revision = job["uses"].split("@")
        self.assertEqual(
            workflow, "pytorch/test-infra/.github/workflows/linux_job_v2.yml"
        )
        self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertEqual(job["with"]["test-infra-ref"], revision)
        self.assertEqual(
            job["with"]["driver-version"],
            "${{ matrix.cuda-version == '13.4' && '615.71.09' || '580.65.06' }}",
        )
        self.assertEqual(
            job["with"]["driver-download-url"],
            "${{ matrix.cuda-version == '13.4' && "
            "'https://download.nvidia.com/XFree86/Linux-x86_64/615.71.09/"
            "NVIDIA-Linux-x86_64-615.71.09.run' || '' }}",
        )

    def test_cuda_probe_rejects_a_different_torch_train(self):
        script = (ROOT / ".ci/scripts/test-cuda-build.sh").read_text()
        probes = [block.split('\n"', 1)[0] for block in script.split('python -c "')[1:]]
        probe = next(block for block in probes if "import torch" in block)
        fake_torch = """
import sys
from types import SimpleNamespace
class Tensor:
    device = 'cuda'
    shape = (10, 10)
    def to(self, device):
        return self
sys.modules['torch'] = SimpleNamespace(
    __version__='test', version=SimpleNamespace(cuda='13.4'),
    cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        current_device=lambda: 0, get_device_name=lambda: 'test',
    ),
    device=lambda name: name, randn=lambda *args: Tensor(),
    mm=lambda x, y: Tensor(),
)
"""
        for expected, succeeds in (("13.4", True), ("13.0", False)):
            with self.subTest(expected=expected):
                result = subprocess.run(
                    [sys.executable, "-c", fake_torch + probe],
                    env={**os.environ, "EXPECTED_CUDA_VERSION": expected},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, succeeds, result.stdout)

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
