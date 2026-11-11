# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
