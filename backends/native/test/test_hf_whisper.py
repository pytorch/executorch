# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Whisper export contracts shared by the native runtime engines.

A small random-weight model exercises native delegation, three-method
packaging, cache initialization, and eager numerics through prefill and
incremental decoding. Runtime operator support is covered by engine tests.
"""

import operator
import os
import tempfile
import unittest

import torch
from executorch.backends.native import get_default_compile_config
from executorch.backends.native.examples.hf.export_hf_whisper import (
    _dump_test_vectors,
    _greedy_transcript,
    _load,
    build_exported_programs,
    example_input_features,
    prompt_ids,
    WhisperDecoder,
)
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_submodules,
)
from executorch.exir.native import to_native
from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
)


class WhisperNativeExportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = WhisperConfig(
            vocab_size=256,
            num_mel_bins=80,
            d_model=64,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            # Two layers also pins cross_kv's multi-output ordering
            # (delegate outputs follow computation order).
            decoder_layers=2,
            decoder_attention_heads=4,
            decoder_ffn_dim=128,
            max_source_positions=50,
            max_target_positions=16,
            decoder_start_token_id=2,
            pad_token_id=0,
            eos_token_id=2,
            bos_token_id=1,
        )
        self.model = WhisperForConditionalGeneration(config).eval()
        self.model.generation_config.forced_decoder_ids = [(1, 10)]
        self.features = example_input_features(config)
        (
            self.encoder,
            self.cross_kv,
            self.decoder,
            self.programs,
        ) = build_exported_programs(self.model, self.features)

    def test_random_weights_loads_checkpoint_generation_config(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.config.save_pretrained(path)
            GenerationConfig(forced_decoder_ids=[(1, 10)]).save_pretrained(path)

            loaded = _load(path, 0, 0, random_weights=True)

        self.assertEqual(prompt_ids(loaded), [2, 10])

    def test_random_weights_warns_when_generation_config_is_missing(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.config.save_pretrained(path)

            with self.assertLogs(
                "executorch.backends.native.examples.hf.export_hf_whisper",
                level="WARNING",
            ) as logs:
                loaded = _load(path, 0, 0, random_weights=True)

        self.assertIn("No generation config found", logs.output[0])
        self.assertEqual(prompt_ids(loaded), [2])

    def test_random_weights_rejects_malformed_generation_config(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.config.save_pretrained(path)
            with open(os.path.join(path, "generation_config.json"), "w") as f:
                f.write("{invalid json")

            with self.assertRaises(OSError):
                _load(path, 0, 0, random_weights=True)

    def test_dump_vectors_without_eos(self):
        self.model.config.eos_token_id = None
        with tempfile.TemporaryDirectory() as path:
            _dump_test_vectors(path, self.model, self.encoder, self.features, 1)

            with open(os.path.join(path, "expected_ids.txt")) as f:
                ids = [int(line) for line in f]
            with open(os.path.join(path, "input_features.bin"), "rb") as f:
                features = f.read()

        self.assertEqual(ids[:2], [2, 10])
        self.assertEqual(len(ids), 3)
        self.assertEqual(features, self.features.contiguous().numpy().tobytes())

    def test_prompt_ids_from_generation_config(self):
        self.assertEqual(prompt_ids(self.model), [2, 10])

    def test_prompt_ids_detects_language_without_shifting_forced_tokens(self):
        self.model.generation_config.forced_decoder_ids = [(1, None), (2, 10)]
        self.model.generation_config.lang_to_id = {"<|en|>": 11, "<|fr|>": 12}
        language = int(self.model.detect_language(input_features=self.features).item())

        self.assertEqual(prompt_ids(self.model, self.features), [2, language, 10])

    def test_prompt_ids_rejects_gaps(self):
        self.model.generation_config.forced_decoder_ids = [(2, 10)]

        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            prompt_ids(self.model, self.features)

    def test_greedy_transcript_respects_token_and_context_limits(self):
        prefix = prompt_ids(self.model)
        capacity = self.model.config.max_target_positions
        for budget in (0, 1, 32):
            with self.subTest(max_new_tokens=budget):
                decoder = WhisperDecoder(self.model).eval()
                transcript = _greedy_transcript(
                    self.encoder,
                    self.cross_kv,
                    decoder,
                    self.features,
                    prefix,
                    set(),
                    budget,
                )

                self.assertEqual(transcript[: len(prefix)], prefix)
                self.assertEqual(len(transcript), min(len(prefix) + budget, capacity))
                unused_start = len(transcript) - 1 if budget else 0
                for layer in decoder.self_cache.layers:
                    self.assertEqual(
                        torch.count_nonzero(layer.keys[:, :, unused_start:]).item(), 0
                    )
                    self.assertEqual(
                        torch.count_nonzero(layer.values[:, :, unused_start:]).item(), 0
                    )

    def test_all_methods_wholly_delegate(self):
        for name, ep in self.programs.items():
            with self.subTest(method=name):
                edge = to_edge_transform_and_lower(
                    ep,
                    transform_passes=get_default_passes(),
                    partitioner=[
                        NativePartitioner(
                            external_constants_tag=None, _serialize_as_ptn=True
                        )
                    ],
                    compile_config=get_default_compile_config(),
                )
                gm = edge.exported_program().graph_module
                self.assertEqual(len(get_lowered_submodules(gm)), 1)
                residual = [
                    node.target
                    for node in gm.graph.nodes
                    if node.op == "call_function"
                    and node.target not in (executorch_call_delegate, operator.getitem)
                ]
                self.assertEqual(residual, [])

    def test_exported_decoder_cache_is_zero_initialized(self):
        buffers = list(self.programs["decoder"].buffers())
        self.assertEqual(len(buffers), 2 * self.model.config.decoder_layers)
        for buffer in buffers:
            self.assertEqual(torch.count_nonzero(buffer).item(), 0)

    def test_exported_programs_match_eager(self):
        with torch.no_grad():
            expected_hidden = self.encoder(self.features)
            actual_hidden = self.programs["encoder"].module()(self.features)
            if isinstance(actual_hidden, (tuple, list)):
                actual_hidden = actual_hidden[0]
            torch.testing.assert_close(actual_hidden, expected_hidden)

            expected_cross = self.cross_kv(expected_hidden)
            actual_cross = self.programs["cross_kv"].module()(expected_hidden)
            self.assertEqual(len(actual_cross), len(expected_cross))
            for actual_t, expected_t in zip(actual_cross, expected_cross):
                torch.testing.assert_close(actual_t, expected_t)

            # Prefill two tokens, then decode single tokens against the
            # live cache; eager and exported modules keep separate caches.
            steps = [
                (torch.tensor([[3, 4]]), torch.tensor([0, 1])),
                (torch.tensor([[5]]), torch.tensor([2])),
                (torch.tensor([[6]]), torch.tensor([3])),
            ]
            exported_decoder = self.programs["decoder"].module()
            for ids, positions in steps:
                expected = self.decoder(ids, positions, *expected_cross)
                actual = exported_decoder(ids, positions, *actual_cross)
                if isinstance(actual, (tuple, list)):
                    actual = actual[0]
                torch.testing.assert_close(actual, expected)

    def test_three_method_package_saves(self):
        # Regression: each method lifts its own scalar constants under the
        # same auto-generated names; the package must still save.
        program = to_native(self.programs)
        self.assertEqual(program.methods, {"encoder", "cross_kv", "decoder"})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "whisper.ptn")
            program.save(path)
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
