# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "cuda.yml").read_text())


def _supported_cuda_versions():
    import runpy

    versions = runpy.run_path(str(ROOT / "install_utils.py"))["SUPPORTED_CUDA_VERSIONS"]
    return [f"{major}.{minor}" for major, minor in versions]


def _matrix_cuda_versions(job):
    matrix = job["strategy"]["matrix"]
    if "include" in matrix:
        return [row["cuda"] for row in matrix["include"]]
    return list(matrix.get("cuda") or matrix["cuda-version"])


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
    def test_build_matrix_covers_supported_cuda_versions(self):
        job = WORKFLOW["jobs"]["test-cuda-builds"]
        self.assertEqual(
            job["strategy"]["matrix"]["cuda-version"], _supported_cuda_versions()
        )
        self.assertEqual(job["with"]["gpu-arch-version"], "${{ matrix.cuda-version }}")

    def test_every_cuda_job_uses_a_supported_version(self):
        supported = _supported_cuda_versions()
        checked = 0
        for workflow in ("cuda.yml", "cuda-windows.yml"):
            jobs = yaml.safe_load(
                (ROOT / ".github" / "workflows" / workflow).read_text()
            )["jobs"]
            for name, job in jobs.items():
                with_ = job.get("with", {})
                version = with_.get("gpu-arch-version")
                if with_.get("gpu-arch-type") != "cuda":
                    continue
                if "${{" in str(version):
                    versions = _matrix_cuda_versions(job)
                else:
                    versions = [version]
                for version in versions:
                    checked += 1
                    with self.subTest(workflow=workflow, job=name, version=version):
                        self.assertIn(version, supported)
        self.assertGreater(checked, 0)

    def test_windows_jobs_cover_every_supported_train(self):
        jobs = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "cuda-windows.yml").read_text()
        )["jobs"]
        for name in (
            "export-model-cuda-windows-artifact",
            "test-model-cuda-windows-e2e",
            "unittest-cuda-runtime-windows",
        ):
            with self.subTest(job=name):
                self.assertEqual(
                    sorted(set(_matrix_cuda_versions(jobs[name]))),
                    sorted(_supported_cuda_versions()),
                )
        export, e2e = (
            jobs[name]["strategy"]["matrix"]["include"]
            for name in (
                "export-model-cuda-windows-artifact",
                "test-model-cuda-windows-e2e",
            )
        )
        self.assertEqual(export, e2e)

    def test_windows_export_image_matches_the_export_job(self):
        build = (ROOT / ".ci" / "docker" / "build.sh").read_text()
        match = re.search(
            r"executorch-ubuntu-22\.04-cuda-windows\)(.*?);;", build, re.S
        )
        self.assertIsNotNone(match)
        image_version = re.search(r"CUDA_VERSION=(\S+)", match.group(1)).group(1)
        self.assertIn(image_version, _supported_cuda_versions())
        jobs = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "cuda-windows.yml").read_text()
        )["jobs"]
        export = jobs["export-model-cuda-windows-artifact"]
        self.assertIn(image_version, _matrix_cuda_versions(export))
        script = export["with"]["script"]
        self.assertIn('!= "${{ matrix.cuda }}" ]; then', script)
        self.assertIn('install_cuda_redist.py --train "${{ matrix.cuda }}"', script)

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

    def test_cuda_runtime_update_precedes_build_and_propagates_failure(self):
        script = WORKFLOW["jobs"]["test-cuda-builds"]["with"]["script"]
        stubs = """
conda() { printf 'CONDA %s\n' "$*"; return "$CONDA_STATUS"; }
source() { printf 'BUILD %s\n' "$*"; printf 'LIBRARY PATH %s\n' "$LD_LIBRARY_PATH"; }
"""
        for version, conda_status in (
            (version, status) for version in ("13.2", "13.4") for status in (0, 1)
        ):
            with self.subTest(version=version, conda_status=conda_status):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        stubs + script.replace("${{ matrix.cuda-version }}", version),
                    ],
                    env={
                        **os.environ,
                        "CONDA_STATUS": str(conda_status),
                        "LD_LIBRARY_PATH": "/cuda/lib64",
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, conda_status, result.stderr)
                expected = [
                    "CONDA install -y -n base -c conda-forge "
                    "libstdcxx-ng=16.2.0 libgcc-ng=16.2.0"
                ]
                if conda_status == 0:
                    expected.extend(
                        (
                            f"BUILD .ci/scripts/test-cuda-build.sh {version}",
                            "LIBRARY PATH /opt/conda/lib:/cuda/lib64",
                        )
                    )
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
            ("13.2", "0", False),
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

    def test_model_e2e_only_uploads_benchmark_results(self):
        self.assertNotIn("test-cuda-pybind", WORKFLOW["jobs"])
        job = WORKFLOW["jobs"]["test-model-cuda-e2e"]
        self.assertNotIn("download-artifact", set(_all_keys(job)))
        self.assertTrue(job["with"]["upload-artifact"].startswith("cuda-bench-"))
        # Exported .pte/.ptd live outside RUNNER_ARTIFACT_DIR; only the
        # benchmark JSON under bench/ is uploaded.
        script = job["with"]["script"]
        self.assertIn('MODEL_DIR="$(mktemp -d "${RUNNER_TEMP:-/tmp}/', script)
        self.assertEqual(1, script.count("RUNNER_ARTIFACT_DIR"))
        self.assertIn('BENCH_RESULTS_DIR="${RUNNER_ARTIFACT_DIR}/bench"', script)
