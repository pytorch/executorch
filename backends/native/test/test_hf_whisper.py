# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Whisper export through the native backend, without a Vulkan device.

Uses a shrunk random-weight Whisper config (same op mix as whisper-tiny,
tiny shapes). Pins the contract the Vulkan engine relies on:

- the encoder, cross-KV projection and static-cache decoder delegate
  wholly to ONE native delegate each, so to_native can package them as a
  three-method .ptn;
- every delegated op either has an ET-VK registry kernel or is rewritten
  at load time by the Vulkan engine (cached self-attention SDPA and its
  index_put_ cache writes by LowerHFAttention's learned-position form,
  cross/encoder SDPA by LowerSDPA; the reinplaced activations
  gelu_/sigmoid_/relu_ map to functional kernels);
- exported SDPA nodes carry no mask or a bool/float mask and are never
  is_causal, which is what the two runtime lowerings require;
- the exported programs match eager numerics across a prefill and
  single-token decode steps against the live KV cache.
"""

from __future__ import annotations

import operator
import os
import subprocess
import tempfile
import unittest
from importlib.resources import as_file, files
from pathlib import Path

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

# Delegated ops the Vulkan engine rewrites before dispatch instead of
# looking them up in the ET-VK registry as-is.
_RUNTIME_REWRITTEN = {
    "aten.scaled_dot_product_attention.default",  # LowerHFAttention/LowerSDPA
    "aten.index_put_.default",  # LowerHFAttention -> llama.update_cache
    "aten.alias_copy.default",  # mask alias consumed/erased by LowerHFAttention
    "aten.gelu_.default",  # functional in-place mapping (reinplace)
    "aten.sigmoid_.default",
    "aten.relu_.default",
    "aten.rms_norm.default",  # LowerRMSNorm -> et_vk.rms_norm (not Whisper)
}

# Every other delegated Whisper op must be in this set of ops registered in
# backends/vulkan/runtime/graph/ops/impl (VK_REGISTER_OP). This is the
# subset Whisper uses, not the full registry: if Transformers starts
# emitting a new op, the test fails until kernel support is confirmed.
_ETVK_REGISTERED = {
    "aten._to_copy.default",
    "aten.add.Tensor",
    "aten.arange.start_step",
    "aten.bitwise_and.Tensor",
    "aten.bmm.default",
    "aten.cat.default",
    "aten.clone.default",
    "aten.convolution.default",
    "aten.embedding.default",
    "aten.expand_copy.default",
    "aten.full.default",
    "aten.index.Tensor",
    "aten.index_select.default",
    "aten.le.Tensor",
    "aten.linear.default",
    "aten.mm.default",
    "aten.mul.Tensor",
    "aten.mul.Scalar",
    "aten.native_layer_norm.default",
    "aten.ne.Scalar",
    "aten.permute_copy.default",
    "aten.repeat.default",
    "aten.slice_copy.Tensor",
    "aten.sub.Tensor",
    "aten.sym_size.int",
    "aten.unsqueeze_copy.default",
    "aten.view_copy.default",
    "et_vk.prepack.default",
    # symbolic int arithmetic on the dynamic token axis
    "add",
    "floordiv",
    "mul",
    "sub",
}


_SYMBOLIC_OPS = {
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.floordiv: "floordiv",
}


def _op_name(target) -> str:
    """Registry-style name for an edge/aten overload or python operator."""
    if target in _SYMBOLIC_OPS:
        return _SYMBOLIC_OPS[target]
    # Only EdgeOpOverload unwraps via _op; a plain OpOverload's _op
    # attribute is an unrelated pybind method.
    op = target._op if type(target).__name__ == "EdgeOpOverload" else target
    name = str(op)
    if name.startswith("torch.ops."):
        name = name[len("torch.ops.") :]
    return name


def _delegated_ops(ep) -> tuple[set[str], set[str], int]:
    """(delegated op names, residual op names, delegate count) for ``ep``."""
    edge = to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[
            NativePartitioner(external_constants_tag=None, _serialize_as_ptn=True)
        ],
        compile_config=get_default_compile_config(),
    )
    gm = edge.exported_program().graph_module
    lowered = get_lowered_submodules(gm)
    delegated: set[str] = set()
    for _, module, _ in lowered:
        for node in module.original_module.graph_module.graph.nodes:
            if node.op == "call_function" and node.target is not operator.getitem:
                delegated.add(_op_name(node.target))
    residual = {
        _op_name(n.target)
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target not in (executorch_call_delegate, operator.getitem)
    }
    return delegated, residual, len(lowered)


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

    def test_both_methods_wholly_delegate_to_vulkan_ops(self):
        for name, ep in self.programs.items():
            with self.subTest(method=name):
                delegated, residual, delegates = _delegated_ops(ep)
                self.assertEqual(delegates, 1)
                self.assertEqual(residual, set())
                unsupported = delegated - _ETVK_REGISTERED - _RUNTIME_REWRITTEN
                self.assertEqual(unsupported, set(), f"{name}: {unsupported}")
                if name != "cross_kv":
                    # The SDPA lowerings are load-bearing in both methods.
                    self.assertIn(
                        "aten.scaled_dot_product_attention.default", delegated
                    )
                if name == "decoder":
                    # Cross K/V arrive as inputs: the decoder projects only
                    # self-attention K/V into its static cache.
                    self.assertIn("aten.index_put_.default", delegated)

    def test_sdpa_nodes_satisfy_lowering_preconditions(self):
        for name, ep in self.programs.items():
            with self.subTest(method=name):
                sdpa_nodes = [
                    n
                    for n in ep.graph_module.graph.nodes
                    if "scaled_dot_product_attention" in str(n.target)
                ]
                if name != "cross_kv":
                    self.assertTrue(sdpa_nodes)
                for node in sdpa_nodes:
                    self.assertFalse(node.kwargs.get("is_causal", False))
                    if len(node.args) > 3 and node.args[3] is not None:
                        # Float additive bias, or the bool keep-mask HF
                        # emits (LowerSDPA converts it at load time).
                        mask_dtype = node.args[3].meta["val"].dtype
                        self.assertIn(
                            mask_dtype,
                            (torch.float32, torch.float16, torch.bool),
                            node.name,
                        )

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

    def test_ggml_runner_matches_eager(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "whisper.ptn"
            to_native(self.programs).save(str(package))
            features = outdir / "input.bin"
            features.write_bytes(self.features.contiguous().numpy().tobytes())
            prompt = outdir / "prompt.txt"
            prefix = prompt_ids(self.model)
            prompt.write_text("".join(f"{token}\n" for token in prefix))
            expected_path = outdir / "expected.txt"
            for budget, cached_graphs in ((1, 1), (32, 0), (32, 1), (32, 2)):
                with self.subTest(budget=budget, cached_graphs=cached_graphs):
                    expected = _greedy_transcript(
                        self.encoder,
                        self.cross_kv,
                        WhisperDecoder(self.model).eval(),
                        self.features,
                        prefix,
                        {2},
                        budget,
                    )
                    expected_path.write_text(
                        "".join(f"{token}\n" for token in expected)
                    )
                    with as_file(files(__package__) / "run_whisper_ggml_cpu") as runner:
                        result = subprocess.run(
                            [
                                str(runner),
                                f"--package={package}",
                                f"--input={features}",
                                f"--prompt={prompt}",
                                f"--expected={expected_path}",
                                "--eos_ids=2",
                                "--cpu_threads=2",
                                f"--max_new_tokens={budget}",
                                f"--max_cached_graphs={cached_graphs}",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn("result:    PASS", result.stdout)

    def test_two_method_package_saves(self):
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
