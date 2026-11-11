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
    OffGraphLastTokenLM,
)
from executorch.extension.llm.cache.hf_attention import register_offgraph_attention
from executorch.extension.llm.cache.reference_cache import LayerPolicy
from executorch.extension.llm.cache.update_and_attend import REGISTRY
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    Gemma3TextConfig,
    GemmaConfig,
    GraniteConfig,
    LlamaConfig,
    MistralConfig,
    PreTrainedModel,
    Qwen2Config,
    Qwen3Config,
)
from transformers.cache_utils import StaticLayer

# Architectures gguf_hf maps; the GGUF test covers each.
_CONFIGS = {
    "llama": LlamaConfig,
    "granite": GraniteConfig,
    "qwen2": Qwen2Config,
    "qwen3": Qwen3Config,
    "gemma": GemmaConfig,
    "gemma2": Gemma2Config,
    "gemma3": Gemma3TextConfig,
}


def _tiny(
    arch: str = "llama", tie_word_embeddings: bool = True, **config_kwargs
) -> PreTrainedModel:
    config = (_CONFIGS | {"mistral": MistralConfig})[arch](
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
        **config_kwargs,
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


def _tiny_muse_glimmer_dir(path: str) -> PreTrainedModel | None:
    """Saves a 4-layer Muse Glimmer config, window 3 on the first three layers,
    to `path` and returns the model, or None where transformers has none."""
    try:
        from transformers import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration
    except ImportError:
        return None
    config = MuseGlimmerConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "sliding_window": 3,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "layer_rope_theta": [10000.0] * 3 + [0],
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
        projector_hidden_size=32,
        out_hidden_size=32,
    )
    config.save_pretrained(path)
    model = MuseGlimmerForConditionalGeneration(config).eval().requires_grad_(False)
    model.config._attn_implementation = "sdpa"
    return model


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
                # Newer transformers' sdpa rejects Gemma 2's attention softcap.
                for model in (reference, loaded):
                    model.config._attn_implementation = "eager"
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

    def test_gguf_loads_the_first_layers_into_a_truncated_model(self) -> None:
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        torch.manual_seed(1)
        reference = _tiny("llama")
        pairs = _as_gguf_tensors(reference, "llama")
        with tempfile.TemporaryDirectory() as path:
            reference.config.save_pretrained(path)
            loaded = _load(path, 1, False, empty_weights=True)

        load_gguf(loaded, "model.gguf", "llama", lambda _: iter(pairs), truncated=True)

        reference.model.layers = reference.model.layers[:1]
        reference.config.num_hidden_layers = 1
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

    def test_offgraph_cache_matches_the_static_cache(self) -> None:
        torch.manual_seed(7)
        model = _tiny()
        register_offgraph_attention("offgraph")
        offgraph_model = copy.deepcopy(model)
        offgraph_model.config._attn_implementation = "offgraph"
        ref = LastTokenLM(model, 32).eval()
        offgraph = OffGraphLastTokenLM(offgraph_model, 32).eval()
        prompts = [[3, 4, 5, 6, 7, 8], [9, 10]]

        samples = golden_samples(offgraph, prompts, 2)
        want = golden_samples(ref, prompts, 2)

        for got, expected in zip(samples.samples, want.samples, strict=True):
            for call, ref_call in zip(got.calls, expected.calls, strict=True):
                self.assertTrue(torch.equal(call.inputs[0], ref_call.inputs[0]))
                self.assertTrue(
                    torch.allclose(
                        call.outputs[0], ref_call.outputs[0], atol=1e-5, rtol=1e-5
                    )
                )
        seq = torch.export.Dim("seq", min=1, max=31)
        ids = torch.tensor([[3, 4, 5]])
        with torch.no_grad():
            ep = torch.export.export(
                offgraph,
                (ids, torch.arange(3)),
                dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                strict=True,
            )
        ops = _ops_except_casts(ep)
        self.assertEqual(ops["kvcache.update_and_attend.default"], 2)
        self.assertNotIn("aten.index_copy_.default", ops)
        self.assertFalse(ep.graph_signature.buffers_to_mutate)

    def test_muse_glimmer_exports_text_only_with_an_offgraph_cache(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            torch.manual_seed(7)
            reference = _tiny_muse_glimmer_dir(path)
            if reference is None:
                self.skipTest("transformers has no Muse Glimmer")
            with self.assertRaisesRegex(ValueError, "only with --offgraph_cache"):
                _load(path, 0, True, False)
            model = _load(path, 0, True, False, offgraph_cache=True)
        missing, unexpected = model.load_state_dict(
            reference.state_dict(), strict=False
        )
        self.assertFalse(missing)
        self.assertTrue(all(".language_model." not in k for k in unexpected))
        offgraph = OffGraphLastTokenLM(model, 32).eval()
        prompts = [[3, 4, 5, 6, 7, 8, 9], [9, 10]]

        samples = golden_samples(offgraph, prompts, 3)

        for sample in samples.samples:
            sequence = []
            for call in sample.calls:
                sequence += call.inputs[0][0].tolist()
                with torch.no_grad():
                    want = reference(torch.tensor([sequence])).logits[:, -1:]
                torch.testing.assert_close(call.outputs[0], want, atol=1e-5, rtol=1e-5)
        self.assertEqual(offgraph.metadata()["get_windows"], [3, 3, 3, 0])
        seq = torch.export.Dim("seq", min=1, max=31)
        with torch.no_grad():
            ep = torch.export.export(
                offgraph,
                (torch.tensor([[3, 4, 5]]), torch.arange(3)),
                dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                strict=False,
            )
        ops = _ops_except_casts(ep)
        self.assertEqual(ops["kvcache.update_and_attend.default"], 4)
        self.assertNotIn("aten.eq.Scalar", ops)

    def test_offgraph_cache_excludes_in_graph_cache_flags(self) -> None:
        for flags in (
            ["--max_context_length", "64"],
            ["--kv_cache_dtype", "fp16"],
        ):
            argv = ["export_hf_llm", "--model", "unused", "--offgraph_cache", *flags]
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", argv):
                with mock.patch.object(export_hf_llm, "_load") as load:
                    with self.assertRaises(SystemExit):
                        export_hf_llm.main()
                load.assert_not_called()

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
        model = _tiny("gemma3")
        if all(
            type(layer) is StaticLayer
            for layer in LastTokenLM(copy.deepcopy(model), 32).cache.layers
        ):
            self.skipTest("this transformers windows static layers with the mask")
        with self.assertRaisesRegex(ValueError, "cannot change the KV dtype"):
            LastTokenLM(model, 32, torch.float16)

    def test_offgraph_cache_windows_sliding_layers_like_hf(self) -> None:
        # Mistral windows every layer; Gemma3 alternates sliding and full.
        register_offgraph_attention("offgraph")
        prompts = [[3, 4, 5, 6, 7, 8, 9], [9, 10]]
        for arch, kwargs in (
            ("mistral", {"sliding_window": 3}),
            (
                "gemma3",
                {
                    "sliding_window": 3,
                    "layer_types": ["sliding_attention", "full_attention"],
                },
            ),
        ):
            with self.subTest(arch=arch):
                torch.manual_seed(7)
                model = _tiny(arch, **kwargs)
                offgraph_model = copy.deepcopy(model)
                offgraph_model.config._attn_implementation = "offgraph"
                offgraph = OffGraphLastTokenLM(offgraph_model, 32).eval()

                samples = golden_samples(offgraph, prompts, 3)

                for sample in samples.samples:
                    # Each call feeds the tokens it appends: the prompt, then one.
                    sequence = []
                    for call in sample.calls:
                        sequence += call.inputs[0][0].tolist()
                        with torch.no_grad():
                            want = model(torch.tensor([sequence])).logits[:, -1:]
                        torch.testing.assert_close(
                            call.outputs[0], want, atol=1e-5, rtol=1e-5
                        )

    def test_offgraph_metadata_gives_each_layer_its_window(self) -> None:
        for model, windows in (
            (_tiny(), [0, 0]),
            (_tiny("mistral", sliding_window=8), [8, 8]),
            (_tiny("mistral", sliding_window=None), [0, 0]),
            (
                _tiny(
                    "gemma3",
                    sliding_window=8,
                    layer_types=["sliding_attention", "full_attention"],
                ),
                [8, 0],
            ),
            (_tiny("qwen2", use_sliding_window=False), [0, 0]),
        ):
            with self.subTest(arch=model.config.model_type):
                wrapper = OffGraphLastTokenLM(model, 32)
                self.assertEqual(
                    wrapper.metadata(),
                    {
                        "get_n_caches": 2,
                        "get_kv_heads": [2, 2],
                        "get_head_dims": [16, 16],
                        "get_windows": windows,
                        "get_max_context_len": 64,
                    },
                )
                self.assertEqual(
                    wrapper._reference.config.layers,
                    [LayerPolicy.ring(w) if w else LayerPolicy.flat() for w in windows],
                )

    def test_offgraph_cache_rejects_other_attention_layer_types(self) -> None:
        model = _tiny()
        model.config.layer_types = ["chunked_attention", "full_attention"]
        with self.assertRaisesRegex(ValueError, "chunked_attention"):
            OffGraphLastTokenLM(model, 32)

    def test_offgraph_cache_scope_uninstalls_its_reference_cache(self) -> None:
        wrapper = OffGraphLastTokenLM(_tiny(), 32)
        with wrapper.cache_scope():
            self.assertIn(wrapper._key, REGISTRY._caches)
        self.assertNotIn(wrapper._key, REGISTRY._caches)
