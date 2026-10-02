# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End to end: export_solo_batching.py's artifact run by run_solo_batching.

Exports the tiny model once, then drives the built runner binary with token-id
prompts and checks its JSON report. The tiny model's weights are random, so
the checks are ones that hold whatever it generates: every generation
completes, the captured decode graph generates what eager decode does,
forwards are shared across sessions, weights load once, and the KV pool grows
with use. Tokens are not compared across batch compositions: a prompt
prefilled in a wider forward runs GEMMs of another shape, and bf16 rounding
can steer greedy decoding elsewhere.

Needs CUDA and the runner (`make muse-glimmer-cuda`); its path may be set with
MUSE_GLIMMER_BATCHING_RUNNER.

    python -m pytest examples/models/muse-glimmer/tests/test_run_solo_batching.py -v
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

import executorch.backends.cuda.quantize_op_dispatch as _quantize_op_dispatch  # noqa: F401
import torch
from executorch.examples.models.muse_glimmer.export.export_solo import (
    load_prequantized_model,
)
from executorch.examples.models.muse_glimmer.export.export_solo_batching import (
    export_batching,
)
from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
    save_checkpoint,
    TINY_CONFIG,
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_RUNNER = os.environ.get(
    "MUSE_GLIMMER_BATCHING_RUNNER",
    os.path.join(_REPO, "cmake-out", "examples", "models", "muse-glimmer", "run_solo_batching"),
)
MAX_STEP = 16
MAX_CELLS = 256
NEW_TOKENS = 8
MIB = 1024 * 1024


class RunSoloBatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if not os.path.exists(_RUNNER):
            raise unittest.SkipTest(f"runner not built: {_RUNNER}")
        cls.tmp = tempfile.mkdtemp()
        ckpt = os.path.join(cls.tmp, "ckpt")
        cls.artifact = os.path.join(cls.tmp, "artifact")
        save_checkpoint(ckpt)
        model, config = load_prequantized_model(ckpt, max_seq_len=TINY_CONFIG.max_seq_len)
        export_batching(
            model, config, cls.artifact, max_step_tokens=MAX_STEP, max_cells=MAX_CELLS
        )
        generator = torch.Generator().manual_seed(0)
        vocab = TINY_CONFIG.vocab_size

        def prompt(length):
            return torch.randint(0, vocab, (length,), generator=generator).tolist()

        # One token, a prefill that runs as decodes (< 5), one prefill, and one
        # wider than a forward so it slices.
        cls.prompts = [prompt(1), prompt(3), prompt(9), prompt(30)]

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def run_prompts(self, prompts, **flags) -> dict:
        prompts_file = os.path.join(self.tmp, "prompts.txt")
        report_file = os.path.join(self.tmp, "report.json")
        with open(prompts_file, "w") as f:
            for tokens in prompts:
                f.write(" ".join(map(str, tokens)) + "\n")
        args = {
            "model_path": os.path.join(self.artifact, "model.pte"),
            "data_path": os.path.join(self.artifact, "aoti_cuda_blob.ptd"),
            "prompt_tokens_file": prompts_file,
            "max_new_tokens": NEW_TOKENS,
            "max_sessions": 4,
            "max_session_tokens": TINY_CONFIG.max_seq_len,
            "kv_initial_capacity": 16,
            # Never a stop token: every generation runs to the limit.
            "eos_id": TINY_CONFIG.vocab_size + 1,
            "report_json": report_file,
        }
        args.update(flags)
        command = [_RUNNER] + [
            f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in args.items()
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        with open(report_file) as f:
            return json.load(f)

    def test_concurrent_generations_complete_in_shared_forwards(self) -> None:
        duplicate = self.prompts[2]
        report = self.run_prompts(self.prompts + [duplicate], max_sessions=5, max_session_tokens=48)
        generations = report["generations"]
        self.assertEqual(len(generations), 5)
        for generation in generations:
            self.assertEqual(generation["finish_reason"], "token_limit")
            self.assertEqual(len(generation["tokens"]), NEW_TOKENS)
            self.assertTrue(
                all(0 <= t < TINY_CONFIG.vocab_size for t in generation["tokens"])
            )
        # Run one after another, every generated token would take a forward of
        # its own; sharing forwards takes far fewer.
        engine = report["engine"]
        self.assertEqual(engine["steps_failed"], 0)
        self.assertEqual(engine["decode_tokens_total"], 5 * (NEW_TOKENS - 1))
        self.assertLess(engine["steps"], engine["decode_tokens_total"])

    def test_captured_decode_graph_generates_what_eager_decode_does(self) -> None:
        prompt = [self.prompts[3]]
        graph = self.run_prompts(prompt, cuda_graph=True)
        eager = self.run_prompts(prompt, cuda_graph=False)
        self.assertEqual(
            graph["generations"][0]["tokens"], eager["generations"][0]["tokens"]
        )

    def test_weights_load_once_whatever_the_session_count(self) -> None:
        one = self.run_prompts(self.prompts[:1], max_sessions=1)
        many = self.run_prompts(self.prompts[:1], max_sessions=4, max_session_tokens=64)
        load = [
            r["gpu"]["used_after_load_bytes"] - r["gpu"]["used_before_load_bytes"]
            for r in (one, many)
        ]
        # The pool is allocated at the first step, not at load, and grows with
        # use: reserving for more sessions costs nothing up front.
        self.assertLess(abs(load[0] - load[1]), 64 * MIB)
        # Loading is the weights once, not once per method, plus a fixed cost.
        self.assertLess(load[0], 2 * one["weights_bytes"] + 256 * MIB)

    def test_kv_pool_grows_with_use(self) -> None:
        report = self.run_prompts(self.prompts, max_session_tokens=48)
        kv = report["kv"]
        tokens = sum(len(p) + NEW_TOKENS for p in self.prompts)
        self.assertGreaterEqual(kv["rows"], kv["cells_in_use"])
        self.assertGreaterEqual(kv["growth_count"], 1)
        # Geometric growth past a 16-row start: at most twice what was needed.
        self.assertLessEqual(kv["rows"], max(16, 2 * (tokens + MAX_STEP)))
        self.assertEqual(kv["allocated_bytes"], kv["rows"] * kv["bytes_per_cell"])
        # Far short of reserving every session's full context up front.
        self.assertLess(kv["rows"], 4 * 48)
