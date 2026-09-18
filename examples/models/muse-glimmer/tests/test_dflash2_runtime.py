# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compiled-runner regressions for four-row verification with a 16-position draft.

Set MUSE_GLIMMER_DFLASH_RUNNER and MUSE_GLIMMER_TOKENIZER to run these tests.
MUSE_GLIMMER_WORKER enables the reset/replay test. An existing fixture exported
by this test can be reused through MUSE_GLIMMER_DFLASH_TEST_ARTIFACT.
"""

import json
import os
import struct
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available()
    and os.environ.get("MUSE_GLIMMER_DFLASH_RUNNER")
    and os.environ.get("MUSE_GLIMMER_TOKENIZER"),
    "Requires CUDA, a built DFlash runner, and the Muse Glimmer tokenizer",
)
class DFlash2RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.artifact = Path(
            os.environ.get("MUSE_GLIMMER_DFLASH_TEST_ARTIFACT", cls.root)
        )
        if cls.artifact == cls.root:
            cls.export_fixture()
        cls.common = [
            f"--model_path={cls.artifact}/model.pte",
            f"--data_path={cls.artifact}/aoti_cuda_blob.ptd",
            f"--tokenizer_path={os.environ['MUSE_GLIMMER_TOKENIZER']}",
            "--bos_id=0",
            # Outside the synthetic vocabulary: these tests exercise capacity.
            "--eos_id=256",
        ]

    @classmethod
    def export_fixture(cls):
        from executorch.examples.models.muse_glimmer.export.export_dflash import (
            _export_dflash_cuda,
        )
        from executorch.examples.models.muse_glimmer.model.dflash_model import (
            DFlashConfig,
            DFlashDraftModel,
            MuseGlimmerWithDFlash,
        )
        from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
        from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
            TINY_CONFIG,
        )

        torch.manual_seed(42)
        tc = replace(TINY_CONFIG, max_seq_len=128, global_attn_cfg="[32,32,32,0]")
        target = MuseGlimmerModel(tc).eval().to(torch.bfloat16)
        target.activation_dtype = torch.bfloat16
        dc = DFlashConfig(
            dim=256,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=64,
            ffn_dim=512,
            vocab_size=256,
            block_size=16,
            target_layers=[2, 6],
            max_seq_len=128,
            sliding_window=32,
            sliding_window_pattern=[True, True],
            conv_kernel_size=2,
            conv_group_size=16,
            selector_rank=16,
            selector_top_k=16,
            mask_token_id=255,
            output_multiplier=0.19611613513,
            final_logit_softcapping=20.0,
        )
        draft = DFlashDraftModel(dc, max_context_length=128).eval().to(torch.bfloat16)

        class PaddedVerification(MuseGlimmerWithDFlash):
            def target_forward_from_embeddings(self, inputs_embeds, input_pos):
                logits, hidden = super().target_forward_from_embeddings(
                    inputs_embeds, input_pos
                )
                # Quantized CUDA exports may retain maximum output sizes.
                padding = (0, 0, 0, 4 - inputs_embeds.shape[1])
                return (
                    torch.nn.functional.pad(logits, padding),
                    torch.nn.functional.pad(hidden, padding),
                )

        with patch(
            "executorch.examples.models.muse_glimmer.model.dflash_model.MuseGlimmerWithDFlash",
            PaddedVerification,
        ):
            _export_dflash_cuda(
                target, tc, draft, dc, None, None, str(cls.root), 128, torch.bfloat16, 0
            )

    def test_graph_transitions_to_single_token_tail(self):
        prompt = self.root / "prompt.bin"
        prompt.write_bytes(struct.pack("<96q", *range(1, 97)))
        outputs = []
        for label, flags in (
            ("graph", ["--cuda_graph=true"]),
            ("reference", ["--cuda_graph=false", "--n_draft=1"]),
        ):
            tokens = self.root / f"{label}.bin"
            result = subprocess.run(
                [os.environ["MUSE_GLIMMER_DFLASH_RUNNER"]]
                + self.common
                + [
                    f"--prompt_tokens_file={prompt}",
                    "--tokens_have_bos=true",
                    "--temperature=0",
                    "--max_new_tokens=32",
                    f"--generated_tokens_file={tokens}",
                ]
                + flags,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout.decode(errors="replace")
            )
            outputs.append(tokens.read_bytes())
            if label == "graph":
                line = next(
                    line
                    for line in result.stdout.decode(errors="replace").splitlines()
                    if line.startswith("DFlashDecodeTiming ")
                )
                timing = json.loads(line.split(" ", 1)[1])
                self.assertGreater(timing["speculative_cycles"], 0)
                self.assertGreater(timing["target_only_cycles"], 0)
                self.assertLessEqual(len(timing["draft_attempts_by_row"]), 3)
        self.assertEqual(len(outputs[0]), 32 * 8)
        self.assertEqual(*outputs)

    def test_rejects_more_than_three_proposals(self):
        result = subprocess.run(
            [os.environ["MUSE_GLIMMER_DFLASH_RUNNER"]]
            + self.common
            + ["--n_draft=4", "--prompt=hello", "--max_new_tokens=1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"n_draft must be at most 3", result.stdout)

    @unittest.skipUnless(os.environ.get("MUSE_GLIMMER_WORKER"), "Requires built worker")
    def test_graph_replay_after_tail_and_new_request(self):
        for enabled in (True, False):
            with self.subTest(cuda_graph=enabled):
                self._check_replay_after_tail(enabled)

    def _check_replay_after_tail(self, enabled):
        generation = {
            "prompt_segments": [{"ids": list(range(1, 96))}],
            "temperature": 0,
            "max_new_tokens": 32,
        }
        requests = [generation, generation]
        result = subprocess.run(
            [os.environ["MUSE_GLIMMER_WORKER"]]
            + self.common
            + ["--max_sessions=1", f"--cuda_graph={str(enabled).lower()}"],
            input="".join(json.dumps(request) + "\n" for request in requests),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        generations = []
        for line in result.stdout.splitlines():
            message = json.loads(line)
            self.assertNotIn("error", message, result.stderr)
            if "generated_token_ids" in message:
                generations.append(message["generated_token_ids"])
        self.assertEqual(len(generations), 2)
        self.assertEqual(len(generations[0]), 32)
        self.assertEqual(*generations)
