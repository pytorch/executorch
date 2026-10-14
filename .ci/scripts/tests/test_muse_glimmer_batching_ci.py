# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".ci/scripts"


class MuseGlimmerBatchingCITest(unittest.TestCase):
    def test_e2e_builds_the_ci_tool_before_running_it(self):
        script = (SCRIPTS / "test_model_e2e.sh").read_text()
        build = script.index("bash .ci/scripts/build_muse_glimmer_batching_bench.sh")
        run = script.index(
            "cmake-out/examples/models/muse-glimmer-batching/muse_glimmer_batching_bench"
        )
        self.assertLess(build, run)
        workflow = (ROOT / ".github/workflows/cuda.yml").read_text()
        unit_job = workflow.split("  unittest-cuda-muse-glimmer-batching:", 1)[1].split(
            "  unittest-cuda-quantized-gemm:", 1
        )[0]
        self.assertIn("bash .ci/scripts/build_muse_glimmer_batching_bench.sh", unit_job)
        self.assertIn("test_muse_glimmer_batching_*.py", unit_job)
        self.assertIn(".ci/scripts/*muse_glimmer_batching*", workflow)

    def test_completion_without_actual_batching_fails_the_smoke_check(self):
        script = (SCRIPTS / "test_model_e2e.sh").read_text()
        checker = script.split('  python - "$BENCH_REPORT" <<\'PY\'\n', 1)[1].split(
            "\nPY\n", 1
        )[0]
        for changed, succeeds in (
            ({}, True),
            ({"max_batch_sessions": 1, "multi_session_steps": 0}, False),
            ({"failed_requests": 1}, False),
            ({"requests": [{"failed": False, "completion_tokens": 63}] * 5}, False),
        ):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                report = {
                    "requests": [{"failed": False, "completion_tokens": 64}] * 5,
                    "failed_requests": 0,
                    "max_batch_sessions": 3,
                    "multi_session_steps": 10,
                    "tokens_per_s": 100.0,
                }
                report.update(changed)
                path = Path(directory) / "report.json"
                path.write_text(json.dumps(report))
                result = subprocess.run(
                    [sys.executable, "-c", checker, str(path)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, succeeds, result.stderr)
