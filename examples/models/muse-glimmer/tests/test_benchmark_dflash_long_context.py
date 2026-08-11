# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from executorch.examples.models.muse_glimmer.dev.benchmark_dflash_long_context import (
    BOS_TOKEN_ID,
    build_runner_command,
    DEFAULT_OUTPUT_TOKENS,
    generate_prompt_files,
    make_prompt_tokens,
    MAX_SEQ_LEN,
    parse_runner_metrics,
    PROMPT_LENGTHS,
    RANDOM_VOCAB_LIMIT,
    run_case,
    token_file_length,
    validate_context_boundary,
    validate_generated_tokens_match,
)


class BenchmarkDFlashLongContextTest(unittest.TestCase):
    def test_prompts_have_exact_lengths_and_common_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            files = generate_prompt_files(
                Path(temp_dir), PROMPT_LENGTHS, seed=17, output_tokens=512
            )

            contents = {length: path.read_bytes() for length, path in files.items()}
            for length, path in files.items():
                self.assertEqual(token_file_length(path), length)
                self.assertEqual(
                    contents[length][:8], BOS_TOKEN_ID.to_bytes(8, "little")
                )
                self.assertEqual(
                    contents[length], contents[PROMPT_LENGTHS[-1]][: length * 8]
                )

    def test_random_payload_excludes_reserved_tokens(self) -> None:
        tokens = make_prompt_tokens(4096, seed=9)
        self.assertEqual(tokens[0], BOS_TOKEN_ID)
        self.assertTrue(all(0 <= token < RANDOM_VOCAB_LIMIT for token in tokens[1:]))

    def test_longest_case_fits_with_transient_verifier(self) -> None:
        validate_context_boundary(PROMPT_LENGTHS[-1], DEFAULT_OUTPUT_TOKENS)
        with self.assertRaisesRegex(ValueError, "exceeds max sequence length"):
            validate_context_boundary(MAX_SEQ_LEN - 1, DEFAULT_OUTPUT_TOKENS)

    def test_runner_commands_share_exact_input_contract(self) -> None:
        common = {
            "model": Path("/tmp/model.pte"),
            "data": Path("/tmp/model.ptd"),
            "tokenizer": Path("/tmp/tokenizer"),
            "prompt": Path("/tmp/prompt.i64"),
            "generated": Path("/tmp/generated.i64"),
            "output_tokens": 512,
        }
        solo = build_runner_command(
            implementation="solo", runner=Path("/tmp/solo"), **common
        )
        dflash = build_runner_command(
            implementation="dflash", runner=Path("/tmp/dflash"), **common
        )

        for command in (solo, dflash):
            self.assertIn("--cuda_graph", command)
            self.assertNotIn("--cuda_graph_target", command)
            self.assertNotIn("--cuda_graph_draft", command)
            self.assertIn("--tokens_have_bos", command)
            self.assertIn("--temperature=0", command)
            self.assertIn("--max_new_tokens=512", command)
            self.assertIn("--ignore_eos", command)
            self.assertIn("--prompt_tokens_file=/tmp/prompt.i64", command)

    def test_run_case_creates_generated_directory_and_overwrites_old_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = root / "fake_runner.py"
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "output = next(a.split('=', 1)[1] for a in sys.argv "
                "if a.startswith('--generated_tokens_file='))\n"
                "Path(output).write_bytes((7).to_bytes(8, 'little', signed=True))\n"
                "print('Prefill: 1 tokens in 1.0 ms (1000.0 tok/s)')\n"
                "print('Decode: 1 tokens in 1.0 ms (1000.0 tok/s)')\n",
                encoding="utf-8",
            )
            runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
            generated = root / "nested" / "generated.i64"
            metrics = run_case(
                [str(runner), f"--generated_tokens_file={generated}"],
                root / "logs" / "runner.log",
                generated,
                expected_output_tokens=1,
                environment=os.environ.copy(),
            )
            self.assertEqual(generated.stat().st_size, 8)
            self.assertEqual(metrics["actual_generated_tokens"], 1)

    def test_run_case_rejects_missing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = root / "fake_runner.py"
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "output = next(a.split('=', 1)[1] for a in sys.argv "
                "if a.startswith('--generated_tokens_file='))\n"
                "Path(output).write_bytes((7).to_bytes(8, 'little', signed=True))\n",
                encoding="utf-8",
            )
            runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
            generated = root / "generated.i64"
            with self.assertRaisesRegex(RuntimeError, "missing required metrics"):
                run_case(
                    [str(runner), f"--generated_tokens_file={generated}"],
                    root / "runner.log",
                    generated,
                    expected_output_tokens=1,
                    environment=os.environ.copy(),
                )

    def test_generated_tokens_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            solo = root / "solo.i64"
            dflash = root / "dflash.i64"
            solo.write_bytes(b"same")
            dflash.write_bytes(b"same")
            validate_generated_tokens_match(solo, dflash)

            dflash.write_bytes(b"different")
            with self.assertRaisesRegex(RuntimeError, "different token sequences"):
                validate_generated_tokens_match(solo, dflash)

    def test_parse_runner_metrics(self) -> None:
        metrics = parse_runner_metrics(
            """
Prefill: 8192 tokens in 100.0 ms (81920.0 tok/s)
Decode: 512 tokens in 2000.0 ms (256.0 tok/s)
DFlashDecodeTiming {"cycles":128,"speculative_cycles":100,"target_only_cycles":28,"draft_execute_ms":312.0,"target_execute_ms":800.0,"draft_attempts_by_row":[128,128,128],"draft_accepts_by_row":[120,100,80]}
PyTorchObserver {"decode_token_per_sec": 256.0, "inference_start_ms": 1000, "first_token_ms": 1100, "inference_end_ms": 3000}
"""
        )
        self.assertEqual(metrics["prompt_tokens"], 8192)
        self.assertEqual(metrics["generated_tokens"], 512)
        self.assertAlmostEqual(metrics["acceptance_percent"], 78.125)
        self.assertEqual(metrics["cycles"], 128)
        self.assertEqual(metrics["draft_tokens"], 384)
        self.assertEqual(metrics["accepted_draft_tokens"], 300)
        self.assertEqual(metrics["draft_execute_ms"], 312.0)
        self.assertEqual(metrics["target_execute_ms"], 800.0)
        self.assertEqual(metrics["ttft_ms"], 100)
        self.assertEqual(metrics["inference_end_to_end_ms"], 2000)
        self.assertEqual(metrics["observer"]["decode_token_per_sec"], 256.0)


if __name__ == "__main__":
    unittest.main()
