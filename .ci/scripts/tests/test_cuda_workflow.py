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

    def test_cuda_builds_take_the_node_driver_on_an_unpinned_v3(self):
        # This job used to pin linux_job_v2 to a test-infra SHA because that was
        # the only ref carrying driver-version / driver-download-url, and it
        # installed R615 for the 13.4 cell. On OSDC the driver belongs to the
        # node and a pod cannot replace it, so the job takes what the node has.
        job = WORKFLOW["jobs"]["test-cuda-builds"]
        self.assertEqual(
            job["uses"], "pytorch/test-infra/.github/workflows/linux_job_v3.yml@main"
        )
        for pin in ("test-infra-ref", "driver-version", "driver-download-url"):
            self.assertNotIn(pin, job["with"])

    def test_cuda134_runtime_update_precedes_build_and_propagates_failure(self):
        script = WORKFLOW["jobs"]["test-cuda-builds"]["with"]["script"]
        stubs = """
conda() { printf 'CONDA %s\n' "$*"; return "$CONDA_STATUS"; }
source() { printf 'BUILD %s\n' "$*"; }
"""
        for version, conda_status in (
            ("12.6", 0),
            ("13.0", 0),
            ("13.4", 0),
            ("13.4", 1),
        ):
            with self.subTest(version=version, conda_status=conda_status):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        stubs + script.replace("${{ matrix.cuda-version }}", version),
                    ],
                    env={**os.environ, "CONDA_STATUS": str(conda_status)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, conda_status, result.stderr)
                expected = []
                if version == "13.4":
                    expected.append(
                        "CONDA install -y -n base -c conda-forge "
                        "libstdcxx-ng=16.2.0 libgcc-ng=16.2.0"
                    )
                if conda_status == 0:
                    expected.append(f"BUILD .ci/scripts/test-cuda-build.sh {version}")
                self.assertEqual(result.stdout.splitlines(), expected)

    def test_cuda_probe_checks_the_result_and_torch_train(self):
        script = (ROOT / ".ci/scripts/test-cuda-build.sh").read_text()
        probes = [block.split('\n"', 1)[0] for block in script.split('python -c "')[1:]]
        probe = next(block for block in probes if "import torch" in block)
        fake_torch = """
import os
import sys
from types import SimpleNamespace
class Tensor:
    device = 'cuda'
    shape = (10, 10)
    def to(self, device):
        return self
    def cpu(self):
        return self
    def __matmul__(self, other):
        return self
def assert_close(actual, expected):
    print('RESULT CHECKED')
    assert os.environ['INVALID_CUDA_RESULT'] == '0', 'CUDA result mismatch'
sys.modules['torch'] = SimpleNamespace(
    __version__='test', version=SimpleNamespace(cuda='13.4'),
    cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        current_device=lambda: 0, get_device_name=lambda: 'test',
    ),
    device=lambda name: name, randn=lambda *args: Tensor(),
    mm=lambda x, y: Tensor(),
    testing=SimpleNamespace(assert_close=assert_close),
)
"""
        for expected, invalid_result, succeeds in (
            ("13.4", "0", True),
            ("13.0", "0", False),
            ("13.4", "1", False),
        ):
            with self.subTest(expected=expected, invalid_result=invalid_result):
                result = subprocess.run(
                    [sys.executable, "-c", fake_torch + probe],
                    env={
                        **os.environ,
                        "EXPECTED_CUDA_VERSION": expected,
                        "INVALID_CUDA_RESULT": invalid_result,
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, succeeds, result.stdout)
                if succeeds:
                    self.assertIn("RESULT CHECKED", result.stdout)

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
