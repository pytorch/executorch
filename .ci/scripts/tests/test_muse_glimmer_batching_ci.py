# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".ci/scripts"
CORRECTNESS_PATHS = (
    ".ci/scripts/build_muse_glimmer_batching_smoke.sh",
    ".ci/scripts/muse_glimmer_batching_smoke.cpp",
    ".ci/scripts/muse_glimmer_batching_workload.h",
    ".ci/scripts/tests/test_muse_glimmer_batching_ci.py",
    ".ci/scripts/tests/test_muse_glimmer_batching_workload.py",
    ".ci/scripts/tests/muse_glimmer_batching_workload_test.cpp",
)


class MuseGlimmerBatchingCITest(unittest.TestCase):
    def test_private_buck_targets_include_the_imported_cuda_packages(self):
        cuda = ROOT / "backends/cuda"
        tree = ast.parse((cuda / "tests/targets.bzl").read_text())
        targets = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fields = {keyword.arg: keyword.value for keyword in node.keywords}
                name = fields.get("name")
                if isinstance(name, ast.Constant) and name.value in (
                    "test_sdpa_varlen_private",
                    "test_private_kv_manifest",
                ):
                    targets[name.value] = fields
        self.assertEqual(len(targets), 2)
        packages = {
            "executorch.backends.cuda.passes.": "//executorch/backends/cuda:cuda_passes",
            "executorch.backends.cuda.triton.kernels.": "//executorch/backends/cuda:triton_kernels",
        }
        for name, fields in targets.items():
            dependencies = ast.literal_eval(fields["deps"])
            for source in ast.literal_eval(fields["srcs"]):
                imports = ast.parse((cuda / "tests" / source).read_text())
                for node in ast.walk(imports):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        for prefix, dependency in packages.items():
                            if node.module.startswith(prefix):
                                with self.subTest(target=name, module=node.module):
                                    self.assertIn(dependency, dependencies)

    def test_imported_kernels_are_in_buck_package(self):
        cuda = ROOT / "backends/cuda"
        tree = ast.parse((cuda / "BUCK").read_text())
        targets = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and any(
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "triton_kernels"
                for keyword in node.value.keywords
            )
        ]
        self.assertEqual(len(targets), 1)
        sources = next(
            ast.literal_eval(keyword.value)
            for keyword in targets[0].keywords
            if keyword.arg == "srcs"
        )
        imports = ast.parse((cuda / "passes/lower_offgraph_kv.py").read_text())
        prefix = "executorch.backends.cuda."
        for node in imports.body:
            if isinstance(node, ast.ImportFrom) and node.module.startswith(
                prefix + "triton.kernels."
            ):
                source = node.module.removeprefix(prefix).replace(".", "/") + ".py"
                with self.subTest(imported_module=node.module):
                    self.assertIn(source, sources)

    def test_production_ring_geometry_is_qualified_separately(self):
        workflow = (ROOT / ".github/workflows/cuda.yml").read_text()
        job = workflow.split("  unittest-cuda-private-kv-mg-ring:", 1)[1].split(
            "  unittest-cuda-runtime:", 1
        )[0]
        self.assertIn("runner: mt-l-x86iavx512-11-125-a100", job)
        self.assertIn("--profile mg-ring", job)
        self.assertIn("'backends/aoti'", job)
        self.assertIn("'extension/llm/batching'", job)
        self.assertIn(
            "PrivateVarlenTest::test_muse_glimmer_and_gemma_head_geometry", job
        )
        self.assertIn(
            "PrivateVarlenTest::test_restored_ring_does_not_read_unwritten_nan_slot",
            job,
        )
        self.assertIn("export PYTHONUNBUFFERED=1", job)
        self.assertIn("-DEXECUTORCH_ENABLE_LOGGING=OFF", job)
        self.assertIn("--target muse_glimmer_runtime_kv_probe --parallel", job)
        self.assertIn('exit "${PRIVATE_KV_STATUS}"', job)
        self.assertIn('"qualification.json", "probe.jsonl", "probe.stderr.log"', job)
        shell = "\n".join(
            line[8:] for line in job.split("      script: |\n", 1)[1].splitlines()
        )
        shell = re.sub(r"\$\{\{[^}]+\}\}", "ci-value", shell)
        result = subprocess.run(["bash", "-n"], input=shell, text=True)
        self.assertEqual(result.returncode, 0)

    def test_runtime_owned_compile_buffer_tests_are_packaged_and_run(self):
        cuda = ROOT / "backends/cuda"
        targets = {}
        for source in (cuda / "BUCK", cuda / "tests/targets.bzl"):
            for node in ast.walk(ast.parse(source.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                fields = {keyword.arg: keyword.value for keyword in node.keywords}
                name = fields.get("name")
                if isinstance(name, ast.Constant):
                    targets[name.value] = fields
        self.assertIn(
            "runtime_owned_kv.py",
            ast.literal_eval(targets["runtime_owned_kv"]["srcs"]),
        )
        self.assertIn(
            ":runtime_owned_kv",
            ast.literal_eval(targets["cuda_backend"]["deps"]),
        )
        self.assertIn(
            "//executorch/backends/cuda:runtime_owned_kv",
            ast.literal_eval(targets["test_runtime_owned_kv"]["deps"]),
        )
        workflow = (ROOT / ".github/workflows/cuda.yml").read_text()
        job = workflow.split("  unittest-cuda-runtime:", 1)[1].split(
            "  unittest-cuda:", 1
        )[0]
        self.assertIn("backends/cuda/tests/test_runtime_owned_kv.py", job)

    def test_noisy_timing_enforcement_requires_explicit_opt_in(self):
        workflow = (ROOT / ".github/workflows/cuda.yml").read_text()
        self.assertIn("      enforce_private_kv_performance:\n", workflow)
        for name, end in (
            ("unittest-cuda-muse-glimmer-batching", "unittest-cuda-quantized-gemm"),
            ("unittest-cuda-private-kv-mg-ring", "unittest-cuda-runtime"),
        ):
            job = workflow.split(f"  {name}:\n", 1)[1].split(f"  {end}:\n", 1)[0]
            self.assertIn('PRIVATE_KV_TIMING_ARGS="--report-performance"', job)
            self.assertIn('"${{ github.event_name }}" == "workflow_dispatch"', job)
            self.assertIn(
                '"${{ inputs.enforce_private_kv_performance }}" == "true"', job
            )

    def test_runtime_columns_probe_runs_in_cuda_ci(self):
        workflow = (ROOT / ".github/workflows/cuda.yml").read_text()
        job = workflow.split("  unittest-cuda-muse-glimmer-batching:", 1)[1].split(
            "  unittest-cuda-quantized-gemm:", 1
        )[0]
        self.assertIn("-DMUSE_GLIMMER_BUILD_RUNTIME_KV_PROBE=ON", job)
        self.assertIn("muse_glimmer_runtime_kv_probe --parallel", job)
        self.assertIn("tests/check_runtime_kv_columns.py", job)
        self.assertIn("-DEXECUTORCH_ENABLE_LOGGING=OFF", job)
        self.assertIn("cmake --build --preset llm-release-cuda-install", job)

    def test_e2e_builds_smoke_before_running_it(self):
        script = (SCRIPTS / "test_model_e2e.sh").read_text()
        build = script.index("bash .ci/scripts/build_muse_glimmer_batching_smoke.sh")
        run = script.index(
            "cmake-out/examples/models/muse-glimmer-batching/muse_glimmer_batching_smoke"
        )
        self.assertLess(build, run)
        # A separate measurement pass may follow this correctness command at tip.
        lines = []
        for line in script[run:].splitlines():
            lines.append(line)
            if not line.endswith("\\"):
                break
        smoke = "\n".join(lines)
        for flag in ("--requests 5", "--prompt_tokens 600", "--max_new_tokens 64"):
            self.assertIn(flag, smoke)
        self.assertIn("--stagger_ms 200", smoke)
        self.assertNotIn("bench", smoke)
        self.assertNotIn("report_json", smoke)
        self.assertNotIn("tokens_per_s", smoke)

    def test_workflow_selects_correctness_tests_and_smoke(self):
        workflow = (ROOT / ".github/workflows/cuda.yml").read_text()
        unit_job = workflow.split("  unittest-cuda-muse-glimmer-batching:", 1)[1].split(
            "  unittest-cuda-quantized-gemm:", 1
        )[0]
        self.assertIn("bash .ci/scripts/build_muse_glimmer_batching_smoke.sh", unit_job)
        self.assertIn(
            'test -x "${BATCHING_BIN_DIR}/muse_glimmer_batching_smoke"', unit_job
        )
        self.assertIn("-p 'test_muse_glimmer_batching_ci.py'", unit_job)
        self.assertIn("-p 'test_muse_glimmer_batching_workload.py'", unit_job)
        self.assertNotIn("test_muse_glimmer_batching_*.py", unit_job)
        filters = workflow.split("  pull_request:\n", 1)[1].split(
            "  workflow_dispatch:", 1
        )[0]
        e2e_job = workflow.split("  test-muse-glimmer-cuda-e2e:", 1)[1]
        for path in CORRECTNESS_PATHS:
            self.assertIn(f"      - {path}\n", filters)
            self.assertIn(f"'{path}'", unit_job)
            self.assertIn(f"'{path}'", e2e_job)
        self.assertNotIn(".ci/scripts/*muse_glimmer_batching*", filters)
        self.assertNotIn(".ci/scripts/tests/*muse_glimmer_batching*", filters)
        for prefix in (
            ".ci/scripts/muse_glimmer_batching",
            ".ci/scripts/build_muse_glimmer_batching",
            ".ci/scripts/tests/test_muse_glimmer_batching",
            ".ci/scripts/tests/muse_glimmer_batching",
        ):
            self.assertNotIn(f"'{prefix}'", unit_job + e2e_job)

    def test_smoke_uses_shared_runtime_and_has_no_measurement_features(self):
        source = (SCRIPTS / "muse_glimmer_batching_smoke.cpp").read_text()
        helper = (SCRIPTS / "muse_glimmer_batching_workload.h").read_text()
        self.assertIn("create_muse_glimmer_backend(config)", source)
        self.assertIn("ci::run_requests(", source)
        self.assertIn("ci::check_batching(evidence)", source)
        self.assertIn("std::make_unique<serving::ServingRuntime>", helper)
        self.assertIn("runtime.generate(", helper)
        self.assertIn("config.default_stop_tokens = {};", helper)
        self.assertIn("runtime->shutdown();", helper)
        self.assertIn("handle.wait();", helper)
        self.assertIn("check_requests_complete(options, result)", helper)
        for forbidden in (
            "bench_utils",
            "SweepShape",
            "TimedExecutor",
            "RequestTiming",
            "report_json",
            "nlohmann",
            "tokens_per_s",
            "ttft",
            "tpot",
            "prefill_ms",
            "decode_ms",
            "total_ms",
            "VRAM",
        ):
            self.assertNotIn(forbidden, source + helper)
        self.assertLess(
            source.index("ci::validate_requests("), source.index("tokenizer.load(")
        )

    def test_smoke_cmake_is_opt_in_and_reuses_runtime(self):
        project = ROOT if (ROOT / "examples").is_dir() else ROOT.parent
        cmake = (project / "examples/models/muse-glimmer/CMakeLists.txt").read_text()
        self.assertRegex(
            cmake,
            r'option\(MUSE_GLIMMER_BUILD_CI_SMOKE\s+"[^"]+"\s+OFF\s*\)',
        )
        self.assertIn(
            "if(MUSE_GLIMMER_BUILD_CI_SMOKE AND NOT MUSE_GLIMMER_BUILD_BATCHING)", cmake
        )
        smoke = cmake.split("    if(MUSE_GLIMMER_BUILD_CI_SMOKE)\n", 1)[1].split(
            "\n    endif()", 1
        )[0]
        self.assertIn(
            "muse_glimmer_batching_smoke PRIVATE muse_glimmer_batching_runtime", smoke
        )
        self.assertIn("LINKER:--export-dynamic", smoke)
        self.assertIn(
            "executorch_target_copy_mlx_metallib(muse_glimmer_batching_smoke)", smoke
        )
        self.assertNotIn("BENCHMARK", smoke)
        self.assertNotIn("batching_bench", smoke)

    def test_owned_shell_and_unit_workflow_syntax(self):
        for script in ("test_model_e2e.sh", "build_muse_glimmer_batching_smoke.sh"):
            result = subprocess.run(["bash", "-n", str(SCRIPTS / script)])
            self.assertEqual(result.returncode, 0)
        workflow = (ROOT / ".github/workflows/cuda.yml").read_text()
        unit_job = workflow.split("  unittest-cuda-muse-glimmer-batching:", 1)[1].split(
            "  unittest-cuda-quantized-gemm:", 1
        )[0]
        body = unit_job.split("      script: |\n", 1)[1]
        shell = "\n".join(line[8:] for line in body.splitlines())
        shell = re.sub(r"\$\{\{[^}]+\}\}", "ci-value", shell)
        result = subprocess.run(["bash", "-n"], input=shell, text=True)
        self.assertEqual(result.returncode, 0)
