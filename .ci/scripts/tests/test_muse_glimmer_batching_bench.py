# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]


class MuseGlimmerBatchingBenchTest(unittest.TestCase):
    def test_workload_and_metrics_without_a_model(self):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "the CI workload tests require a C++ compiler")
        root = SCRIPTS.parents[1]
        project = root if (root / "examples").is_dir() else root.parent
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            binary = temporary / "bench_test"
            # Stub only the unused multimodal/tensor boundary, just as the
            # lower helper test does; serving declarations remain real.
            multimodal = (
                temporary / "executorch/extension/llm/runner/multimodal_input.h"
            )
            multimodal.parent.mkdir(parents=True)
            multimodal.write_text(
                "#pragma once\n#include <cstdint>\n#include <vector>\n"
                "namespace executorch::extension::llm {\n"
                "struct MultimodalInput {\n"
                "  explicit MultimodalInput(std::vector<std::uint64_t>) {}\n"
                "};\n}\n"
            )
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-Wno-deprecated-declarations",
                    "-pthread",
                    "-I",
                    str(temporary),
                    "-I",
                    str(project.parent),
                    str(SCRIPTS / "tests/muse_glimmer_batching_bench_test.cpp"),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("tests passed", result.stdout)

    def test_e2e_benchmark_follows_smoke_before_server_setup(self):
        script = (SCRIPTS / "test_model_e2e.sh").read_text()
        smoke = script.index("/muse_glimmer_batching_smoke \\\n")
        build = script.index("  bash .ci/scripts/build_muse_glimmer_batching_bench.sh")
        invocation = script.index("/muse_glimmer_batching_bench \\\n")
        checker = script.index('  python - "$BENCH_REPORT"')
        server = script.index(
            "  pip install -r examples/llm_server/python/requirements.txt"
        )
        self.assertLess(smoke, build)
        self.assertLess(build, invocation)
        self.assertLess(invocation, checker)
        self.assertLess(checker, server)
        self.assertIn('--report_json "$BENCH_REPORT"', script[invocation:checker])

    def test_e2e_report_checker(self):
        script = (SCRIPTS / "test_model_e2e.sh").read_text()
        checker = script.split("  python - \"$BENCH_REPORT\" <<'PY'\n", 1)[1]
        checker = checker.split("\nPY\n", 1)[0]
        report = {
            "requests": [{"failed": False, "completion_tokens": 64}] * 5,
            "failed_requests": 0,
            "max_batch_sessions": 3,
            "multi_session_steps": 10,
            "tokens_per_s": 100.0,
        }
        for changed, succeeds in (
            ({}, True),
            ({"max_batch_sessions": 1, "multi_session_steps": 0}, False),
            ({"max_batch_sessions": 1}, False),
            ({"multi_session_steps": 0}, False),
            ({"failed_requests": 1}, False),
            ({"requests": [{"failed": False, "completion_tokens": 63}] * 5}, False),
            ({"requests": [{"failed": True, "completion_tokens": 64}] * 5}, False),
            ({"requests": report["requests"][:4]}, False),
        ):
            with self.subTest(
                changed=changed
            ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                path.write_text(json.dumps(report | changed))
                result = subprocess.run(
                    [sys.executable, "-", str(path)],
                    input=checker,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, succeeds, result.stderr)

    def test_workflow_explicit_benchmark_discovery_build_and_gates(self):
        workflow = (SCRIPTS.parents[1] / ".github/workflows/cuda.yml").read_text()
        unit = workflow.split("  unittest-cuda-muse-glimmer-batching:\n", 1)[1]
        unit = unit.split("\n  unittest-cuda-quantized-gemm:", 1)[0]
        commands = (
            'test -x "${BATCHING_BIN_DIR}/muse_glimmer_batching_smoke"',
            "python -m unittest discover -s .ci/scripts/tests "
            "-p 'test_muse_glimmer_batching_bench.py' -v",
            'bash .ci/scripts/build_muse_glimmer_batching_bench.sh "${BATCHING_BIN_DIR}"',
            'test -x "${BATCHING_BIN_DIR}/muse_glimmer_batching_bench"',
        )
        positions = [unit.index(command) for command in commands]
        self.assertEqual(positions, sorted(positions))
        e2e = workflow.split("  test-muse-glimmer-cuda-e2e:\n", 1)[1]
        for path in (
            ".ci/scripts/build_muse_glimmer_batching_bench.sh",
            ".ci/scripts/muse_glimmer_batching_bench.cpp",
            ".ci/scripts/muse_glimmer_batching_bench_utils.h",
            ".ci/scripts/tests/test_muse_glimmer_batching_bench.py",
            ".ci/scripts/tests/muse_glimmer_batching_bench_test.cpp",
        ):
            self.assertIn(
                f"      - {path}\n", workflow.split("  workflow_dispatch:")[0]
            )
            gate = f"contains(needs.changed-files.outputs.changed-files, '{path}')"
            self.assertIn(gate, unit.split("    uses:")[0])
            self.assertIn(gate, e2e.split("    uses:")[0])

    def test_benchmark_uses_shared_workload_control(self):
        source = (SCRIPTS / "muse_glimmer_batching_bench.cpp").read_text()
        utils = (SCRIPTS / "muse_glimmer_batching_bench_utils.h").read_text()
        self.assertIn('#include "muse_glimmer_batching_workload.h"', source)
        self.assertIn('#include "muse_glimmer_batching_workload.h"', utils)
        for shared in (
            "validate_requests",
            "make_runtime",
            "submit_request",
            "run_requests",
        ):
            self.assertIn(f"muse_glimmer_ci::{shared}(", source)
        for duplicate in ("struct Inflight", "prompt_ids(", "rt.generate("):
            self.assertNotIn(duplicate, source)
        self.assertNotIn("struct BatchEvidence", utils)
        self.assertNotIn("inline void check_session_capacity", utils)
        self.assertIn("evidence_.observe(batch)", source)
        for callback in ("submitted", "text", "terminal"):
            self.assertIn(f"callbacks.{callback} =", source)

    def test_build_enables_ci_target_and_propagates_failures(self):
        for failure in ("none", "configure", "build"):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                log = root / "commands"
                cmake = root / "cmake"
                cmake.write_text(
                    '#!/bin/bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n'
                    'if [[ "$1" == "--build" ]]; then\n'
                    '  [[ "$FAIL_AT" != "build" ]]\n'
                    'else\n  [[ "$FAIL_AT" != "configure" ]]\nfi\n'
                )
                cmake.chmod(0o755)
                build_dir = root / "build with spaces"
                result = subprocess.run(
                    [
                        "bash",
                        str(SCRIPTS / "build_muse_glimmer_batching_bench.sh"),
                        str(build_dir),
                    ],
                    env={
                        **os.environ,
                        "PATH": f"{root}:{os.environ['PATH']}",
                        "COMMAND_LOG": str(log),
                        "FAIL_AT": failure,
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode == 0, failure == "none", result.stderr
                )
                commands = log.read_text().splitlines()
                self.assertIn("-DMUSE_GLIMMER_BUILD_CI_BENCHMARK=ON", commands[0])
                self.assertIn("-DMUSE_GLIMMER_BUILD_BATCHING=ON", commands[0])
                self.assertIn(str(build_dir), commands[0])
                self.assertEqual(len(commands), 1 if failure == "configure" else 2)
                if len(commands) == 2:
                    self.assertIn("--target muse_glimmer_batching_bench", commands[1])
