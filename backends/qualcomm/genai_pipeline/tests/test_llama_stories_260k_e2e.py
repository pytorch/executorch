# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""x86 output parity coverage for the stories260k GenAI Pipeline.

Runs on the x86 host only: both the pipeline and the legacy ``llama.py`` run
with ``--enable-x86-64``, executing the x86 runner against the QNN SDK's x86
libraries. No device is involved, so this does not cover the on-device path.

It checks that the pipeline's ``prompt_eval`` generations match legacy
``llama.py`` and that its ``sqnr_eval`` SQNR is at least ``_MIN_SQNR``.

Requires ``QNN_SDK_ROOT`` and ``LLAMA_ARTIFACTS``, and an x86 build
(``QNN_X86_BUILD``, default ``build-x86`` under the ExecuTorch root); skipped when
the variables are unset.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_EXECUTORCH_ROOT = Path(__file__).resolve().parents[4]
_REQUIRED_ENV = ("QNN_SDK_ROOT", "LLAMA_ARTIFACTS")
_PIPELINE_RESULT_MARKER = "__EXECUTORCH_PIPELINE_RESULT__="
# Greedy text from a 260k model can match even with misordered inputs, so the
# logits are also held to an SQNR threshold.
_MIN_SQNR = 15


def _assert_x86_prerequisites() -> None:
    qnn_sdk = Path(os.environ["QNN_SDK_ROOT"]).resolve()
    artifacts = Path(os.environ["LLAMA_ARTIFACTS"]).resolve()
    build = Path(
        os.environ.get("QNN_X86_BUILD", str(_EXECUTORCH_ROOT / "build-x86"))
    ).resolve()

    assert qnn_sdk.is_dir(), f"QNN_SDK_ROOT is not a directory: {qnn_sdk}"
    assert artifacts.is_dir(), f"LLAMA_ARTIFACTS is not a directory: {artifacts}"
    for filename in (
        "stories260K.pt",
        "params.json",
        "tokenizer.model",
        "tokenizer.bin",
    ):
        path = artifacts / filename
        assert path.is_file(), f"missing stories260k artifact: {path}"

    qnn_x86_lib = qnn_sdk / "lib" / "x86_64-linux-clang"
    assert qnn_x86_lib.is_dir(), f"missing QNN x86 SDK lib directory: {qnn_x86_lib}"
    assert build.is_dir(), f"QNN x86 build directory does not exist: {build}"

    runner = build / "examples/qualcomm/oss_scripts/llama/qnn_llama_runner"
    assert runner.is_file(), f"missing qnn_llama_runner: {runner}"
    adaptors = tuple(build.rglob("PyQnnManagerAdaptor*.so"))
    assert adaptors, f"missing PyQnnManagerAdaptor*.so under {build}"


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


@unittest.skipUnless(
    all(os.getenv(name) for name in _REQUIRED_ENV),
    "requires QNN_SDK_ROOT and LLAMA_ARTIFACTS",
)
class TestLlamaStories260kPipelineParity(unittest.TestCase):
    def setUp(self):
        _assert_x86_prerequisites()
        self.artifacts = Path(os.environ["LLAMA_ARTIFACTS"]).resolve()
        self.build = Path(
            os.environ.get("QNN_X86_BUILD", str(_EXECUTORCH_ROOT / "build-x86"))
        ).resolve()
        self.soc = os.environ.get("QNN_X86_SOC", "SM8650")

    def _child_environment(self):
        qnn_lib = (
            Path(os.environ["QNN_SDK_ROOT"]).resolve() / "lib" / "x86_64-linux-clang"
        )
        pythonpath = [str(_EXECUTORCH_ROOT / "src"), str(_EXECUTORCH_ROOT)]
        if os.environ.get("PYTHONPATH"):
            pythonpath.append(os.environ["PYTHONPATH"])
        ld_library_path = [str(qnn_lib), str(self.build / "lib")]
        if os.environ.get("LD_LIBRARY_PATH"):
            ld_library_path.append(os.environ["LD_LIBRARY_PATH"])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(ld_library_path)
        return environment

    def _run(self, command):
        completed = subprocess.run(
            command,
            cwd=_EXECUTORCH_ROOT,
            env=self._child_environment(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "child command failed\nstdout:\n{}\nstderr:\n{}".format(
                completed.stdout, completed.stderr
            ),
        )
        return completed

    def test_pipeline_matches_legacy_decoded_output_and_sqnr_threshold(self):
        pipeline_script = (
            "import json; "
            "from executorch.backends.qualcomm.genai_pipeline import cli; "
            "args = cli.build_parser().parse_args(); "
            "args.model = args.model.lower(); "
            "result = cli.run_pipeline(args); "
            'print("__EXECUTORCH_PIPELINE_RESULT__=" + json.dumps({'
            '"generations": result.results["prompt_eval"].generations, '
            '"sqnr": result.results["sqnr_eval"].eval_results["sqnr"]}))'
        )

        with tempfile.TemporaryDirectory(prefix="stories260k-pipeline-") as tmp:
            root = Path(tmp)
            pipeline_artifacts = root / "pipeline"
            legacy_artifacts = root / "legacy"
            pipeline_command = [
                sys.executable,
                "-c",
                pipeline_script,
                "--model",
                "stories260k",
                "--soc",
                self.soc,
                "--artifact-dir",
                str(pipeline_artifacts),
                "--checkpoint",
                str(self.artifacts / "stories260K.pt"),
                "--params",
                str(self.artifacts / "params.json"),
                "--tokenizer-model",
                str(self.artifacts / "tokenizer.model"),
                "--tokenizer-bin",
                str(self.artifacts / "tokenizer.bin"),
                "--prompt",
                "Once",
                "--temperature",
                "0",
                "--model-mode",
                "hybrid",
                "--prefill-ar-len",
                "32",
                "--max-seq-len",
                "128",
                "--max-context-len",
                "128",
                "--calib-tasks",
                "wikitext",
                "--calib-limit",
                "1",
                "--enable-x86-64",
                "--build-folder",
                str(self.build),
                "--eval-methods",
                "prompt_eval",
                "sqnr_eval",
            ]
            pipeline_process = self._run(pipeline_command)
            marker_lines = [
                line
                for line in pipeline_process.stdout.splitlines()
                if line.startswith(_PIPELINE_RESULT_MARKER)
            ]
            self.assertEqual(
                len(marker_lines),
                1,
                "pipeline result marker missing or repeated\nstdout:\n{}".format(
                    pipeline_process.stdout
                ),
            )
            pipeline_output = json.loads(
                marker_lines[0][len(_PIPELINE_RESULT_MARKER) :]
            )
            self.assertIsInstance(pipeline_output["generations"], list)
            # Shown on success with -rP or -s, to track the SQNR against the threshold.
            print(
                f"stories260k sqnr_eval SQNR: {pipeline_output['sqnr']:.2f} "
                f"(threshold {_MIN_SQNR})"
            )
            self.assertGreaterEqual(pipeline_output["sqnr"], _MIN_SQNR)

            legacy_command = [
                sys.executable,
                str(_EXECUTORCH_ROOT / "examples/qualcomm/oss_scripts/llama/llama.py"),
                "--artifact",
                str(legacy_artifacts),
                "--build_folder",
                str(self.build),
                "--soc_model",
                self.soc,
                "--checkpoint",
                str(self.artifacts / "stories260K.pt"),
                "--params",
                str(self.artifacts / "params.json"),
                "--tokenizer_model",
                str(self.artifacts / "tokenizer.model"),
                "--tokenizer_bin",
                str(self.artifacts / "tokenizer.bin"),
                "--prompt",
                "Once",
                "--temperature",
                "0",
                "--decoder_model",
                "stories260k",
                "--model_mode",
                "hybrid",
                "--prefill_ar_len",
                "32",
                "--max_seq_len",
                "128",
                "--max_context_len",
                "128",
                "--calib_tasks",
                "wikitext",
                "--calib_limit",
                "1",
                "--enable_x86_64",
            ]
            self._run(legacy_command)

            legacy_output_path = legacy_artifacts / "outputs" / "outputs.txt"
            self.assertTrue(
                legacy_output_path.is_file(),
                f"legacy output was not created: {legacy_output_path}",
            )
            legacy_output = legacy_output_path.read_bytes().decode()
            self.assertEqual(
                [
                    _normalize_line_endings(text)
                    for text in pipeline_output["generations"]
                ],
                [_normalize_line_endings(legacy_output)],
            )
