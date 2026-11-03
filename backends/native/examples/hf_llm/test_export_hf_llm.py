# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import re
import sys
import tempfile
import unittest
from unittest import mock

import gguf
import torch
from executorch.backends.native.examples.hf_llm import export_hf_llm
from executorch.backends.native.examples.hf_llm.export_hf_llm import (
    _DECODE_STEPS,
    _FLOAT_TOL,
    _load,
    _PROMPTS,
    _quantize_8da4w,
    _QUANTIZED_TOL,
    _sample_prompts,
    golden_samples,
    LastTokenLM,
    load_gguf,
)
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    Gemma3TextConfig,
    GemmaConfig,
    GraniteConfig,
    LlamaConfig,
    PreTrainedModel,
    Qwen2Config,
    Qwen3Config,
)

_CONFIGS = {
    "llama": LlamaConfig,
    "granite": GraniteConfig,
    "qwen2": Qwen2Config,
    "qwen3": Qwen3Config,
    "gemma": GemmaConfig,
    "gemma2": Gemma2Config,
    "gemma3": Gemma3TextConfig,
}


def _tiny(arch: str = "llama", tie_word_embeddings: bool = True) -> PreTrainedModel:
    config = _CONFIGS[arch](
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
    model.config._attn_implementation = "sdpa"
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    return model


def _as_gguf_tensors(
    model: PreTrainedModel, arch: str
) -> list[tuple[str, torch.Tensor]]:
    """The model's weights as llama.cpp's convert_hf_to_gguf.py writes them:
    standard GGUF names, Llama's Q/K rows permuted, Gemma's norms offset by 1."""
    config = model.config
    gguf_arch = next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch)
    tensor_map = gguf.get_tensor_name_map(gguf_arch, config.num_hidden_layers)
    pairs = []
    if arch == "llama":
        pairs.append(("rope_freqs.weight", torch.ones(config.head_dim // 2)))
    for fqn, weight in model.state_dict().items():
        if fqn == "lm_head.weight" and config.tie_word_embeddings:
            continue
        heads = {
            "q_proj": config.num_attention_heads,
            "k_proj": config.num_key_value_heads,
        }.get(fqn.rsplit(".", 2)[-2])
        if arch in ("llama", "granite") and heads is not None:
            weight = (
                weight.reshape(heads, 2, weight.shape[0] // heads // 2, -1)
                .swapaxes(1, 2)
                .reshape(weight.shape)
            )
        if arch.startswith("gemma") and fqn.endswith("norm.weight"):
            weight = weight + 1
        name = tensor_map.get_name(fqn, try_suffixes=(".weight", ".bias"))
        pairs.append((name, weight.clone()))
    return pairs


_CASTS = (
    "aten._assert_tensor_metadata.default",
    "aten._to_copy.default",
    "aten.to.dtype",
)


def _ops_except_casts(ep: torch.export.ExportedProgram) -> collections.Counter:
    return collections.Counter(
        str(n.target)
        for n in ep.graph.nodes
        if n.op == "call_function" and str(n.target) not in _CASTS
    )


class ExportHfLlmTest(unittest.TestCase):
    def test_gguf_weights_reproduce_the_model(self) -> None:
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        cases = [(arch, True) for arch in _CONFIGS] + [("llama", False)]
        for arch, tie in cases:
            with self.subTest(arch=arch, tie_word_embeddings=tie):
                torch.manual_seed(1)
                reference = _tiny(arch, tie)
                for weight in reference.parameters():
                    weight.uniform_(-0.5, 0.5)
                torch.manual_seed(2)
                loaded = _tiny(arch, tie)
                pairs = _as_gguf_tensors(reference, arch)

                load_gguf(loaded, "model.gguf", arch, lambda _, p=pairs: iter(p))

                with torch.no_grad():
                    got, want = loaded(ids).logits, reference(ids).logits
                # Gemma's (w + 1) - 1 rounds; everything else is moved exactly.
                if arch.startswith("gemma"):
                    self.assertTrue(torch.allclose(got, want, atol=1e-4, rtol=0))
                else:
                    self.assertTrue(torch.equal(got, want))

    def test_gguf_rejects_unknown_and_missing_weights(self) -> None:
        pairs = _as_gguf_tensors(_tiny(), "llama")
        cases = {
            "maps to no llama parameter": pairs
            + [("blk.9.attn_q.weight", torch.zeros(64, 64))],
            "missing": [pair for pair in pairs if pair[0] != "blk.1.ffn_up.weight"],
            "GGUF shape": [
                (name, torch.zeros(256, 64) if name == "blk.0.ffn_up.weight" else t)
                for name, t in pairs
            ],
        }
        for message, case in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    load_gguf(_tiny(), "model.gguf", "llama", lambda _, c=case: iter(c))

    def test_gguf_keeps_a_vocabulary_trimmed_to_the_tokenizer(self) -> None:
        model = _tiny()
        pairs = [
            (name, t[:120] if name == "token_embd.weight" else t)
            for name, t in _as_gguf_tensors(_tiny(), "llama")
        ]

        load_gguf(model, "model.gguf", "llama", lambda _: iter(pairs))

        self.assertEqual(model.lm_head.weight.shape, (120, 64))

    def test_gguf_loads_into_a_model_built_without_weights(self) -> None:
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        for tie in (True, False):
            with self.subTest(tie_word_embeddings=tie):
                torch.manual_seed(1)
                reference = _tiny("llama", tie)
                pairs = _as_gguf_tensors(reference, "llama")
                with tempfile.TemporaryDirectory() as path:
                    reference.config.save_pretrained(path)
                    loaded = _load(path, 0, False, empty_weights=True)
                self.assertTrue(all(p.is_meta for p in loaded.parameters()))

                load_gguf(loaded, "model.gguf", "llama", lambda _, p=pairs: iter(p))

                self.assertFalse(
                    any(t.is_meta for t in [*loaded.parameters(), *loaded.buffers()])
                )
                with torch.no_grad():
                    got, want = loaded(ids).logits, reference(ids).logits
                self.assertTrue(torch.equal(got, want))

    def test_gguf_derives_head_dim_when_config_has_none(self) -> None:
        torch.manual_seed(1)
        reference = _tiny()
        loaded = _tiny()
        loaded.config.head_dim = None
        pairs = _as_gguf_tensors(reference, "llama")

        load_gguf(loaded, "model.gguf", "llama", lambda _: iter(pairs))

        self.assertTrue(
            torch.equal(
                loaded.model.layers[0].self_attn.q_proj.weight,
                reference.model.layers[0].self_attn.q_proj.weight,
            )
        )

    def test_fp16_kv_cache_exports_position_explicit_graph(self) -> None:
        torch.manual_seed(7)
        model = _tiny()
        fp32 = LastTokenLM(model, 32).eval()
        fp16 = LastTokenLM(copy.deepcopy(model), 32, torch.float16).eval()
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        positions = torch.arange(ids.shape[1])
        seq = torch.export.Dim("seq", min=1, max=31)

        with torch.no_grad():
            ep32, ep = [
                torch.export.export(
                    module,
                    (ids, positions),
                    dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                    strict=True,
                )
                for module in (fp32, fp16)
            ]
            exported = ep.module()
            ref = fp32(ids, positions)
            got = exported(ids, positions)
            next_id = ref.argmax(dim=-1).reshape(1, 1)
            next_ref = fp32(next_id, torch.tensor([6]))
            next_got = exported(next_id, torch.tensor([6]))

        self.assertTrue(torch.allclose(ref, got, atol=1e-2, rtol=2e-2))
        self.assertTrue(torch.allclose(next_ref, next_got, atol=1e-2, rtol=2e-2))
        caches = {
            name: ep.state_dict.get(name, ep.constants.get(name))
            for name in ep.graph_signature.buffers
            if re.search(r"(key|value)_cache_\d+$", name)
        }
        self.assertTrue(caches)
        self.assertEqual({t.dtype for t in caches.values()}, {torch.float16})
        self.assertEqual(_ops_except_casts(ep), _ops_except_casts(ep32))

    def test_golden_samples_teacher_force_eager_argmax(self) -> None:
        torch.manual_seed(7)
        model = _tiny()
        prompts = [[3, 4, 5, 6, 7, 8], [9, 10]]
        wrapper = LastTokenLM(model, 32).eval()

        samples = golden_samples(wrapper, prompts, 2)

        ref = LastTokenLM(copy.deepcopy(model), 32).eval()
        for ids, sample in zip(prompts, samples.samples, strict=True):
            ref.cache.reset()
            want = [
                (torch.tensor([ids]), torch.arange(len(ids))),
                (None, torch.tensor([len(ids)])),
                (None, torch.tensor([len(ids) + 1])),
            ]
            self.assertEqual(len(sample.calls), len(want))
            previous = None
            for call, (input_ids, positions) in zip(sample.calls, want):
                if input_ids is None:
                    input_ids = previous.argmax(dim=-1)
                self.assertEqual(call.method, "forward")
                self.assertTrue(torch.equal(call.inputs[0], input_ids))
                self.assertTrue(torch.equal(call.inputs[1], positions))
                with torch.no_grad():
                    previous = ref(input_ids, positions)
                self.assertTrue(torch.equal(call.outputs[0], previous))
        self.assertFalse(any(layer.keys.any() for layer in wrapper.cache.layers))

    def test_samples_flag_needs_a_count_in_range_and_an_output(self) -> None:
        for flags in (
            ["--samples", str(len(_PROMPTS) + 1), "--output", "m.ptn"],
            ["--samples", "-1", "--output", "m.ptn"],
            ["--samples", "1"],
        ):
            argv = ["export_hf_llm", "--model", "unused", *flags]
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", argv):
                with mock.patch.object(export_hf_llm, "_load") as load:
                    with self.assertRaises(SystemExit):
                        export_hf_llm.main()
                load.assert_not_called()

    def test_sample_prompts_must_fit_with_their_decode_steps(self) -> None:
        tokenizer = mock.Mock()
        tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        lengths = [len(p.split()) for p in _PROMPTS[:2]]
        longest = max(lengths)

        with mock.patch.object(
            export_hf_llm.AutoTokenizer, "from_pretrained", return_value=tokenizer
        ):
            prompts = _sample_prompts("m", 2, longest + 1, longest + _DECODE_STEPS)
            self.assertEqual(prompts, [list(range(n)) for n in lengths])
            for max_seq_length, cache_len in (
                (longest, 64),
                (64, longest + _DECODE_STEPS - 1),
            ):
                with self.assertRaisesRegex(ValueError, "do not fit"):
                    _sample_prompts("m", 2, max_seq_length, cache_len)

    def test_golden_samples_tolerance_follows_weight_quantization(self) -> None:
        source = _tiny()
        gguf_float = _tiny()
        pairs = _as_gguf_tensors(source, "llama")
        load_gguf(gguf_float, "model.gguf", "llama", lambda _: iter(pairs))
        quantized = _tiny()
        _quantize_8da4w(quantized, group_size=32, embedding_group_size=32)

        for model, tol in (
            (source, _FLOAT_TOL),
            (gguf_float, _FLOAT_TOL),
            (quantized, _QUANTIZED_TOL),
        ):
            samples = golden_samples(LastTokenLM(model, 16).eval(), [[3, 4]], 0)
            self.assertEqual(samples.tol, tol)

    def test_fp16_kv_cache_rejects_sliding_window_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot change the KV dtype"):
            LastTokenLM(_tiny("gemma3"), 32, torch.float16)
