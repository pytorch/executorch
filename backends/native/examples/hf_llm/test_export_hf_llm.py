# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import io
import re
import sys
import tempfile
import unittest
from unittest import mock

import gguf
import torch
from executorch.backends.native import get_default_compile_config
from executorch.backends.native.custom_ops import gated_delta_rule_op
from executorch.backends.native.examples.hf_llm import export_hf_llm
from executorch.backends.native.examples.hf_llm.export_hf_llm import (
    _check_attention_args,
    _DECODE_STEPS,
    _FLOAT_TOL,
    _layer_mask,
    _load,
    _PROMPTS,
    _quantize_8da4w,
    _QUANTIZED_TOL,
    _sample_prompts,
    export_prefill,
    golden_samples,
    LastTokenLM,
    load_gguf,
    native_gdn_rules,
    OffGraphLastTokenLM,
    prefill_calls,
)
from executorch.backends.native.passes import FuseRoPEPass
from executorch.backends.native.passes._utils import _resolve_aten
from executorch.exir import to_edge
from executorch.exir.native import to_native
from executorch.extension.llm.cache.hf_attention import register_offgraph_attention
from executorch.extension.llm.cache.reference_cache import LayerPolicy
from executorch.extension.llm.cache.update_and_attend import REGISTRY
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from torchao.utils import TorchAOBaseTensor
from transformers import (
    AutoModelForCausalLM,
    DeepseekV32Config,
    Gemma2Config,
    Gemma3Config,
    Gemma3TextConfig,
    Gemma4TextConfig,
    Gemma4UnifiedTextConfig,
    GemmaConfig,
    GptOssConfig,
    GraniteConfig,
    Llama4TextConfig,
    LlamaConfig,
    MistralConfig,
    PreTrainedModel,
    Qwen2Config,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3Config,
    ZayaConfig,
)
from transformers.masking_utils import (
    and_masks,
    bidirectional_mask_function,
    blockwise_overlay,
    causal_mask_function,
    chunked_causal_mask_function,
    or_masks,
    packed_sequence_mask_function,
    sdpa_mask,
    sliding_window_bidirectional_mask_function,
    sliding_window_causal_mask_function,
)
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen35moe

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


_IDS = [[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]


def _tiny_qwen35() -> PreTrainedModel:
    """A tiny Qwen3.5: three gated DeltaNet layers, then full attention, with
    more value heads than key heads and interleaved M-RoPE, as the real ones."""
    torch.manual_seed(5)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    model = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
    model.config._attn_implementation = "sdpa"
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    return model


def _tiny_qwen35moe() -> PreTrainedModel:
    """`_tiny_qwen35` with a mixture of four experts, two per token, and a
    gated shared expert, loaded as export loads a model."""
    torch.manual_seed(5)
    config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    with tempfile.TemporaryDirectory() as path:
        config.save_pretrained(path)
        return _load(path, 0, True, empty_weights=False).requires_grad_(False)


def _hf_greedy_logits(
    model: PreTrainedModel, ids: list[int], decode_steps: int
) -> list[torch.Tensor]:
    """transformers' own logits for `ids` in one call, then `decode_steps`
    greedy steps, on its dynamic cache: no export wrapper involved."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=True)
        cache, logits = out.past_key_values, [out.logits[:, -1:]]
        for step in range(decode_steps):
            out = model(
                input_ids=logits[-1][:, -1].argmax(-1, keepdim=True),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([len(ids) + step]),
            )
            logits.append(out.logits[:, -1:])
    return logits


def _one_token_logits(
    run, ids: list[int], reference: list[torch.Tensor]
) -> list[torch.Tensor]:
    """`run`'s last-position logits for `ids` fed one token per call from
    position 0, then for the greedy tokens `reference` chose."""
    with torch.no_grad():
        for position, token in enumerate(ids):
            logits = run(torch.tensor([[token]]), torch.tensor([position]))
        got = [logits]
        for step, previous in enumerate(reference[:-1]):
            token = previous[:, -1].argmax(-1, keepdim=True)
            got.append(run(token, torch.tensor([len(ids) + step])))
    return got


def _random_gemma(arch: str, **config) -> PreTrainedModel:
    """A tiny Gemma with a sliding-window layer, then a full-attention one."""
    torch.manual_seed(3)
    model = _tiny(arch, layer_types=["sliding_attention", "full_attention"], **config)
    for weight in model.parameters():
        weight.uniform_(-0.5, 0.5)
    return model


def _random_gemma4(
    config_cls, randomize: bool = True, **moe: object
) -> PreTrainedModel:
    """A tiny Gemma 4 text decoder shaped as the 12B: a sliding-window layer,
    then a global layer whose values are its keys, with a wider head, one KV
    head and proportional RoPE, and capped final logits."""
    torch.manual_seed(7)
    extra = {"hidden_size_per_layer_input": 0} if config_cls is Gemma4TextConfig else {}
    config = config_cls(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=3,
        max_position_embeddings=64,
        final_logit_softcapping=30.0,
        **extra,
        **moe,
    )
    model = AutoModelForCausalLM.from_config(
        config, experts_implementation="batched_mm" if moe else None
    )
    model = model.eval().requires_grad_(False)
    model.config._attn_implementation = "sdpa"
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    if randomize:
        for weight in model.parameters():
            weight.uniform_(-0.5, 0.5)
        for i, layer in enumerate(model.model.layers):
            layer.layer_scalar.fill_(0.5 + 0.25 * i)
    return model


# HF Gemma 4 FQN suffix -> llama.cpp's tensor name, from its converter.
_GEMMA4_GGUF = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "pre_feedforward_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
    "layer_scalar": "layer_output_scale.weight",
    "router.proj.weight": "ffn_gate_inp.weight",
    "router.scale": "ffn_gate_inp.scale",
    "router.per_expert_scale": "ffn_down_exps.scale",
    "experts.gate_up_proj": "ffn_gate_up_exps.weight",
    "experts.down_proj": "ffn_down_exps.weight",
    "pre_feedforward_layernorm_2.weight": "pre_ffw_norm_2.weight",
    "post_feedforward_layernorm_1.weight": "post_ffw_norm_1.weight",
    "post_feedforward_layernorm_2.weight": "post_ffw_norm_2.weight",
}
_GEMMA4_MOE = {
    "enable_moe_block": True,
    "num_experts": 4,
    "top_k_experts": 2,
    "moe_intermediate_size": 32,
}


def _gemma4_gguf_tensors(model: PreTrainedModel) -> list[tuple[str, torch.Tensor]]:
    """The model's weights as llama.cpp's converter writes a Gemma 4 GGUF:
    norms as stored, the output tied to the embedding, a derived rope_freqs."""
    pairs = [
        ("token_embd.weight", model.model.embed_tokens.weight.clone()),
        ("output_norm.weight", model.model.norm.weight.clone()),
        ("rope_freqs.weight", torch.ones(16)),
    ]
    for fqn, weight in model.state_dict().items():
        layer, _, suffix = fqn.removeprefix("model.layers.").partition(".")
        if fqn.startswith("model.layers.") and suffix in _GEMMA4_GGUF:
            pairs.append((f"blk.{layer}.{_GEMMA4_GGUF[suffix]}", weight.clone()))
    return pairs


def _eager_logits(model: PreTrainedModel, ids: torch.Tensor) -> torch.Tensor:
    """Logits of one uncached pass through transformers' eager attention."""
    eager = copy.deepcopy(model)
    eager.config._attn_implementation = "eager"
    with torch.no_grad():
        return eager(ids, use_cache=False).logits


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


def _k_quant_pairs(
    model: PreTrainedModel, generator: torch.Generator
) -> list[tuple[str, torch.Tensor]]:
    """The model's GGUF tensors with every matrix replaced by random Q4_K
    blocks; d = 2^-12 and dmin = 2^-9 keep the weights at HF-init magnitude."""
    pairs = []
    for name, tensor in _as_gguf_tensors(model, "llama"):
        if tensor.dim() == 2:
            rows, blocks = tensor.shape[0], tensor.shape[1] // 256
            raw = torch.randint(0, 256, (rows, blocks, 144), generator=generator)
            raw = raw.to(torch.uint8)
            raw[..., 0:2] = torch.tensor([0x00, 0x0C], dtype=torch.uint8)
            raw[..., 2:4] = torch.tensor([0x00, 0x18], dtype=torch.uint8)
            tensor = ExportableGGUFTensor.from_raw(
                raw.reshape(rows, -1), "q4_k", torch.float32
            )
        pairs.append((name, tensor))
    return pairs


_CASTS = (
    "aten._assert_tensor_metadata.default",
    "aten._to_copy.default",
    "aten.to.dtype",
)


def _constant_consumers(
    ep: torch.export.ExportedProgram,
) -> dict[str, list[str]]:
    """Each parameter, buffer or constant the graph reads -> the ops reading it."""
    sig = ep.graph_signature
    names = {
        **sig.inputs_to_parameters,
        **sig.inputs_to_buffers,
        **sig.inputs_to_lifted_tensor_constants,
    }
    return {
        names[n.name]: sorted(str(user.target) for user in n.users)
        for n in ep.graph.nodes
        if n.op == "placeholder" and n.name in names
    }


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

    def test_gemma4_gguf_weights_reproduce_the_model(self) -> None:
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        for config_cls in (Gemma4TextConfig, Gemma4UnifiedTextConfig):
            with self.subTest(config_cls.__name__):
                reference = _random_gemma4(config_cls)
                loaded = _random_gemma4(config_cls, randomize=False)
                pairs = _gemma4_gguf_tensors(reference)

                load_gguf(loaded, "model.gguf", "gemma4", lambda _, p=pairs: iter(p))

                self.assertEqual(
                    [layer.layer_scalar.item() for layer in loaded.model.layers],
                    [0.5, 0.75],
                )
                with torch.no_grad():
                    got, want = loaded(ids).logits, reference(ids).logits
                self.assertTrue(torch.equal(got, want))

    def test_gemma4_moe_gguf_weights_reproduce_the_model(self) -> None:
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        reference = _random_gemma4(Gemma4TextConfig, **_GEMMA4_MOE)
        loaded = _random_gemma4(Gemma4TextConfig, randomize=False, **_GEMMA4_MOE)
        pairs = _gemma4_gguf_tensors(reference)
        self.assertIn("blk.1.ffn_gate_up_exps.weight", dict(pairs))

        load_gguf(loaded, "model.gguf", "gemma4", lambda _: iter(pairs))

        with torch.no_grad():
            got, want = loaded(ids).logits, reference(ids).logits
        self.assertTrue(torch.equal(got, want))

    def test_gguf_buffers_it_lacks_keep_their_initial_values(self) -> None:
        pairs = [
            pair
            for pair in _gemma4_gguf_tensors(_random_gemma4(Gemma4TextConfig))
            if pair[0] != "blk.1.layer_output_scale.weight"
        ]
        loaded = _random_gemma4(Gemma4TextConfig, randomize=False)

        load_gguf(loaded, "model.gguf", "gemma4", lambda _: iter(pairs))

        self.assertEqual(
            [layer.layer_scalar.item() for layer in loaded.model.layers], [0.5, 1.0]
        )

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

    def test_gguf_loads_the_text_decoder_of_a_multimodal_config(self) -> None:
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        torch.manual_seed(1)
        reference = _tiny("gemma3")
        config = Gemma3Config(
            text_config=reference.config.to_dict(),
            vision_config={"hidden_size": 16, "num_hidden_layers": 1},
        )
        pairs = _as_gguf_tensors(reference, "gemma3")
        with tempfile.TemporaryDirectory() as path:
            config.save_pretrained(path)
            loaded = _load(path, 0, False, empty_weights=True)
            random = _load(path, 0, True, empty_weights=False)

        for model in (loaded, random):
            self.assertEqual(type(model), type(reference))
        load_gguf(loaded, "model.gguf", "gemma3", lambda _: iter(pairs))

        with torch.no_grad():
            got, want = loaded(ids).logits, reference(ids).logits
        self.assertTrue(torch.allclose(got, want, atol=1e-4, rtol=0))

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

    def test_qwen35_rotary_fuses_into_one_rope_per_projection(self) -> None:
        # Text positions fill all of M-RoPE's rows, so transformers' rotary
        # and apply rotate the queries and keys by one rope each.
        rope = {
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        }
        for modeling, rotary_cls, config_cls in (
            (qwen35, qwen35.Qwen3_5TextRotaryEmbedding, qwen35.Qwen3_5TextConfig),
            (
                qwen35moe,
                qwen35moe.Qwen3_5MoeTextRotaryEmbedding,
                qwen35moe.Qwen3_5MoeTextConfig,
            ),
        ):
            with self.subTest(rotary_cls.__name__):
                rotary = rotary_cls(config_cls(head_dim=32, rope_parameters=rope))

                class Rotate(torch.nn.Module):
                    apply_rope = staticmethod(modeling.apply_rotary_pos_emb)

                    def __init__(self, rotary=rotary):
                        super().__init__()
                        self.rotary = rotary

                    def forward(self, q, k, position_ids):
                        cos, sin = self.rotary(q, position_ids)
                        return self.apply_rope(q, k, cos, sin)

                model = Rotate().eval()
                inputs = (
                    torch.randn(1, 4, 5, 32),
                    torch.randn(1, 2, 5, 32),
                    torch.arange(5).reshape(1, 5) + 7,
                )
                fused = (
                    to_edge(
                        torch.export.export(model, inputs, strict=True),
                        compile_config=get_default_compile_config(),
                    )
                    .transform([FuseRoPEPass()])
                    .exported_program()
                )
                ops = collections.Counter(
                    _resolve_aten(n.target)
                    for n in fused.graph.nodes
                    if n.op == "call_function"
                )
                self.assertEqual(ops[torch.ops.native.rope.default], 2)
                for table_op in (
                    torch.ops.aten.cos.default,
                    torch.ops.aten.sin.default,
                    torch.ops.aten.where.self,
                    torch.ops.aten.slice_scatter.default,
                ):
                    self.assertEqual(ops[table_op], 0)
                torch.testing.assert_close(fused.module()(*inputs), model(*inputs))

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
        # Lowering fails unless the whole graph, the cache op included, forms
        # one native delegate.
        to_native(ep)

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
        selected = OffGraphLastTokenLM(model, 32, selected=True).eval()
        ids = torch.tensor([[3, 4, 5]])
        with torch.no_grad():
            with selected.cache_scope():
                got = selected(ids, torch.arange(3), torch.tensor([2]))
            with offgraph.cache_scope():
                offgraph.reset_cache()
                want = offgraph(ids, torch.arange(3))
        torch.testing.assert_close(got, want)

    def test_offgraph_selected_logits_match_the_last_token(self) -> None:
        torch.manual_seed(7)
        register_offgraph_attention("offgraph")
        model = _tiny()
        model.config._attn_implementation = "offgraph"
        last = OffGraphLastTokenLM(model, 32).eval()
        selected = OffGraphLastTokenLM(model, 32, selected=True).eval()
        prompts = [[3, 4, 5, 6, 7, 8], [9, 10]]

        want = golden_samples(last, prompts, 2)
        got = golden_samples(selected, prompts, 2)

        for sample, expected in zip(got.samples, want.samples, strict=True):
            for call, ref in zip(sample.calls, expected.calls, strict=True):
                ids = call.inputs[0]
                self.assertTrue(
                    torch.equal(call.inputs[2], torch.tensor([ids.shape[1] - 1]))
                )
                self.assertTrue(torch.equal(call.outputs[0], ref.outputs[0]))
        self.assertEqual(selected.metadata()["get_logits_to_keep_mode"], 2)
        self.assertEqual(last.metadata()["get_logits_to_keep_mode"], 1)
        seq = torch.export.Dim("seq", min=1, max=31)
        rows = torch.export.Dim("rows", min=1, max=31)
        with torch.no_grad():
            ep = torch.export.export(
                selected,
                (torch.tensor([[3, 4, 5]]), torch.arange(3), torch.tensor([0, 2])),
                dynamic_shapes={
                    "input_ids": {1: seq},
                    "cache_position": {0: seq},
                    "logits_to_keep": {0: rows},
                },
                strict=True,
            )
            with selected.cache_scope():
                logits = ep.module()(
                    torch.tensor([[3, 4, 5]]), torch.arange(3), torch.tensor([0, 2])
                )
        self.assertEqual(tuple(logits.shape), (1, 2, 128))
        self.assertEqual(len(ep.graph_signature.user_inputs), 3)

    def test_selected_logits_need_an_offgraph_cache(self) -> None:
        argv = ["export_hf_llm", "--model", "unused", "--logits_to_keep", "selected"]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(export_hf_llm, "_load") as load:
                with self.assertRaises(SystemExit):
                    export_hf_llm.main()
        load.assert_not_called()

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

    def test_cache_writes_index_the_cache_position_input(self) -> None:
        seq = torch.export.Dim("seq", min=1, max=31)
        for arch, kv_dtype in (
            ("llama", torch.float32),
            ("gemma2", torch.float32),
            ("gemma3", torch.float16),
        ):
            with self.subTest(arch=arch, kv_dtype=kv_dtype):
                torch.manual_seed(7)
                model = _tiny(arch)
                fp32 = LastTokenLM(model, 32).eval()
                wrapper = LastTokenLM(copy.deepcopy(model), 32, kv_dtype).eval()
                ids = torch.tensor([[3, 4, 5, 6]])
                with torch.no_grad():
                    ep = torch.export.export(
                        wrapper,
                        (ids, torch.arange(4)),
                        dynamic_shapes={
                            "input_ids": {1: seq},
                            "cache_position": {0: seq},
                        },
                        strict=True,
                    )
                    exported = ep.module()
                    for step_ids, positions in (
                        (ids, torch.arange(4)),
                        (torch.tensor([[9]]), torch.tensor([4])),
                        (ids, torch.arange(4)),
                    ):
                        self.assertTrue(
                            torch.allclose(
                                fp32(step_ids, positions),
                                exported(step_ids, positions),
                                atol=1e-2,
                                rtol=2e-2,
                            )
                        )
                writes = [
                    n
                    for n in ep.graph.nodes
                    if n.target == torch.ops.aten.index_copy_.default
                ]
                self.assertEqual(len(writes), 2 * model.config.num_hidden_layers)
                for n in writes:
                    self.assertEqual(n.args[2].name, "cache_position")

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
                        "get_logits_to_keep_mode": 1,
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

    def test_offgraph_cache_refuses_per_layer_head_shapes(self) -> None:
        model = _random_gemma4(Gemma4TextConfig)
        with self.assertRaisesRegex(NotImplementedError, "per layer"):
            OffGraphLastTokenLM(model, 32)

    def test_offgraph_cache_scope_uninstalls_its_reference_cache(self) -> None:
        wrapper = OffGraphLastTokenLM(_tiny(), 32)
        with wrapper.cache_scope():
            self.assertIn(wrapper._key, REGISTRY._caches)
        self.assertNotIn(wrapper._key, REGISTRY._caches)

    def test_cache_layers_with_more_state_than_keys_and_values_are_refused(
        self,
    ) -> None:
        # DeepSeek V3.2's sparse attention caches its indexer's keys too.
        config = DeepseekV32Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=16,
            index_topk=4,
            index_head_dim=16,
            index_n_heads=2,
            max_position_embeddings=64,
        )
        model = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
        model.generation_config.cache_implementation = "static"

        with self.assertRaisesRegex(NotImplementedError, "StaticIndexedLayer"):
            LastTokenLM(model, 16)

    def test_hybrid_cache_layers_are_refused_before_the_export_wrapper(self) -> None:
        # The wrapper would turn a hybrid sliding-window layer into a plain
        # StaticLayer, dropping its linear-attention state.
        for layer_types, sliding_window, refused in (
            (["hybrid"], None, "LinearAttentionAndStaticFullAttentionLayer"),
            (
                ["hybrid_sliding"],
                4,
                "LinearAttentionAndStaticSlidingWindowAttentionLayer",
            ),
        ):
            with self.subTest(layer_types=layer_types):
                config = ZayaConfig(
                    vocab_size=128,
                    hidden_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=16,
                    moe_intermediate_size=32,
                    num_experts=2,
                    router_hidden_size=16,
                    max_position_embeddings=64,
                    layer_types=layer_types,
                    sliding_window=sliding_window,
                )
                model = AutoModelForCausalLM.from_config(config)
                model = model.eval().requires_grad_(False)
                model.generation_config.cache_implementation = "static"
                with self.assertRaisesRegex(NotImplementedError, refused):
                    LastTokenLM(model, 16)

    def test_linear_attention_models_export_one_token_calls_with_state_buffers(
        self,
    ) -> None:
        wrapper = LastTokenLM(_tiny_qwen35(), 32).eval()
        self.assertEqual(wrapper.tokens_per_call, 1)
        with torch.no_grad():
            ep = torch.export.export(
                wrapper,
                (
                    torch.zeros((1, 1), dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                ),
                strict=True,
            )
        signature = ep.run_decompositions({}).graph_signature
        mutated = set(signature.buffers_to_mutate.values())
        states = {name for name in mutated if "_state_" in name}
        # Three linear-attention layers, each with a conv and a recurrent state.
        self.assertEqual(len(states), 6)
        self.assertTrue(states <= set(signature.non_persistent_buffers))

    def test_linear_attention_matches_transformers_after_a_restart(self) -> None:
        # Independent of the export wrapper: transformers' own dynamic cache,
        # the prompt in one call. The wrapper and its exported program take one
        # token per call, and the second prompt restarts at position 0 with no
        # reset call: the graph zeroes the states there.
        model = _tiny_qwen35()
        wrapper = LastTokenLM(copy.deepcopy(model), 32).eval()
        with torch.no_grad():
            ep = torch.export.export(
                wrapper,
                (
                    torch.zeros((1, 1), dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                ),
                strict=True,
            )
        for run in (wrapper, ep.module()):
            for ids in ([3, 4, 5, 6, 7, 8, 9], [10, 11, 12]):
                with self.subTest(run=type(run).__name__, ids=ids):
                    want = _hf_greedy_logits(model, ids, 3)
                    got = _one_token_logits(run, ids, want)
                    for a, b in zip(got, want, strict=True):
                        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_moe_experts_export_as_gathers_and_batched_matmuls(self) -> None:
        model = _tiny_qwen35moe()
        self.assertEqual(model.config._experts_implementation, "batched_mm")
        wrapper = LastTokenLM(copy.deepcopy(model), 32).eval()
        with torch.no_grad():
            ep = torch.export.export(
                wrapper,
                (
                    torch.zeros((1, 1), dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                ),
                strict=True,
            ).run_decompositions({})
        targets = [n.target for n in ep.graph.nodes if n.op == "call_function"]
        aten = torch.ops.aten
        # Each layer gathers its gate_up and down experts by routed id.
        self.assertEqual(targets.count(aten.index.Tensor), 8)
        self.assertEqual(targets.count(aten.bmm.default), 8)
        for run in (wrapper, ep.module()):
            for ids in ([3, 4, 5, 6, 7], [8, 9]):
                with self.subTest(run=type(run).__name__, ids=ids):
                    want = _hf_greedy_logits(model, ids, 3)
                    got = _one_token_logits(run, ids, want)
                    for a, b in zip(got, want, strict=True):
                        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_native_gdn_rule_is_one_op_per_layer_and_call(self) -> None:
        # transformers' own rules and dynamic cache, the prompt in one call,
        # against the wrapper's 8-token and one-token calls on native's op.
        model = _tiny_qwen35()
        ids = list(range(3, 22))
        want = _hf_greedy_logits(model, ids, 2)
        with native_gdn_rules(model), torch.no_grad():
            wrapper = LastTokenLM(copy.deepcopy(model), 32).eval()
            programs = {
                size: torch.export.export(
                    wrapper,
                    (torch.zeros((1, size), dtype=torch.long), torch.arange(size)),
                    strict=True,
                )
                for size in (1, 8)
            }
            start = 0
            for size in (8, 8, 1, 1, 1):
                chunk = torch.tensor([ids[start : start + size]])
                logits = wrapper(chunk, torch.arange(start, start + size))
                start += size
        torch.testing.assert_close(logits, want[0], atol=1e-4, rtol=1e-4)
        for name, program in programs.items():
            with self.subTest(method=name):
                targets = [
                    n.target for n in program.graph.nodes if n.op == "call_function"
                ]
                # One per linear-attention layer; no ATen rule is left.
                self.assertEqual(targets.count(gated_delta_rule_op), 3)
                self.assertNotIn(torch.ops.aten.cumsum.default, targets)

    def test_native_gdn_rules_restore_transformers_rules(self) -> None:
        model = _tiny_qwen35()
        module = sys.modules[type(model.model.layers[0].linear_attn).__module__]
        rules = (
            module.torch_chunk_gated_delta_rule,
            module.torch_recurrent_gated_delta_rule,
        )
        with native_gdn_rules(model):
            self.assertIsNot(module.torch_chunk_gated_delta_rule, rules[0])
        self.assertEqual(
            (
                module.torch_chunk_gated_delta_rule,
                module.torch_recurrent_gated_delta_rule,
            ),
            rules,
        )

    def test_prefill_calls_take_full_calls_then_the_rest(self) -> None:
        self.assertEqual(prefill_calls(1100, 512), [512, 512, 76])
        self.assertEqual(prefill_calls(513, 512), [512, 1])
        self.assertEqual(prefill_calls(3, 0), [1, 1, 1])

    def test_prefill_takes_any_token_count_and_matches_transformers(self) -> None:
        # transformers' own rules and dynamic cache, the prompt in one call,
        # against prefill calls of 8, 8 and 3 tokens on native's op, then the
        # exported prefill on calls of 2, 5 and 8 tokens against eager.
        model = _tiny_qwen35()
        ids = list(range(3, 22))
        want = _hf_greedy_logits(model, ids, 2)
        with native_gdn_rules(model), torch.no_grad():
            wrapper = LastTokenLM(copy.deepcopy(model), 32).eval()
            prefill = export_prefill(wrapper, 8)
            forward = torch.export.export(
                wrapper, (torch.zeros((1, 1), dtype=torch.long), torch.arange(1))
            )
            self.assertEqual(
                set(prefill.graph_signature.buffers_to_mutate.values()),
                set(forward.graph_signature.buffers_to_mutate.values()),
            )
            start = 0
            for size in prefill_calls(len(ids), 8):
                chunk = torch.tensor([ids[start : start + size]])
                logits = wrapper(chunk, torch.arange(start, start + size))
                start += size
            torch.testing.assert_close(logits, want[0], atol=1e-4, rtol=1e-4)
            for size in (2, 5, 8):
                with self.subTest(tokens=size):
                    first = (torch.tensor([ids[:size]]), torch.arange(size))
                    fresh = LastTokenLM(copy.deepcopy(model), 32).eval()
                    torch.testing.assert_close(prefill.module()(*first), fresh(*first))
            targets = [n.target for n in prefill.graph.nodes if n.op == "call_function"]
            self.assertEqual(targets.count(gated_delta_rule_op), 3)

    def test_golden_samples_call_prefill_then_forward(self) -> None:
        with native_gdn_rules(model := _tiny_qwen35()):
            wrapper = LastTokenLM(model, 32).eval()
            (sample,) = golden_samples(wrapper, [list(range(3, 14))], 2, 8).samples
        methods = [call.method for call in sample.calls]
        self.assertEqual(methods, ["prefill", "prefill", "forward", "forward"])
        positions = torch.cat([call.inputs[1] for call in sample.calls])
        self.assertTrue(torch.equal(positions, torch.arange(13)))

    def test_golden_samples_refuse_an_empty_prompt(self) -> None:
        with native_gdn_rules(model := _tiny_qwen35()):
            wrappers = [LastTokenLM(_tiny(), 32), LastTokenLM(model, 32)]
            for wrapper in wrappers:
                with self.subTest(recurrent=bool(wrapper.tokens_per_call)):
                    with self.assertRaisesRegex(ValueError, "at least one token"):
                        golden_samples(wrapper.eval(), [[3, 4], []], 1, 8)

    def test_prefill_tokens_needs_the_native_rule_and_a_fitting_count(self) -> None:
        for flags in (
            ["--gdn_op", "aten", "--prefill_tokens", "8"],
            ["--prefill_tokens", "1"],
            ["--prefill_tokens", "200"],
        ):
            argv = ["export_hf_llm", "--model", "unused", *flags]
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", argv):
                with mock.patch.object(export_hf_llm, "_load") as load:
                    with self.assertRaises(SystemExit):
                        export_hf_llm.main()
                load.assert_not_called()

    def test_golden_samples_feed_linear_attention_prompts_one_token_per_call(
        self,
    ) -> None:
        wrapper = LastTokenLM(_tiny_qwen35(), 32).eval()
        samples = golden_samples(wrapper, [[3, 4, 5], [6]], 2)
        for ids, sample in zip([[3, 4, 5], [6]], samples.samples, strict=True):
            self.assertEqual(len(sample.calls), len(ids) + 2)
            for position, call in enumerate(sample.calls):
                self.assertEqual(call.inputs[0].shape, (1, 1))
                self.assertTrue(torch.equal(call.inputs[1], torch.tensor([position])))
                if position < len(ids):
                    self.assertEqual(call.inputs[0].item(), ids[position])
                else:
                    previous = sample.calls[position - 1].outputs[0]
                    self.assertEqual(
                        call.inputs[0].item(), previous[0, -1].argmax().item()
                    )
        self.assertFalse(
            any(
                state.any()
                for name, state in wrapper.named_buffers()
                if "_state_" in name
            )
        )

    def test_chunked_attention_layers_match_eager_past_a_chunk(self) -> None:
        ids = torch.tensor(_IDS)
        torch.manual_seed(3)
        config = Llama4TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            intermediate_size_mlp=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            moe_layers=[],
            no_rope_layers=[1, 0],
            attention_chunk_size=3,
        )
        model = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
        model.config._attn_implementation = "sdpa"
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        self.assertEqual(
            model.config.layer_types, ["chunked_attention", "full_attention"]
        )
        unchunked = copy.deepcopy(model)
        unchunked.config.attention_chunk_size = 64
        self.assertFalse(
            torch.allclose(
                _eager_logits(unchunked, ids), _eager_logits(model, ids), atol=1e-3
            )
        )
        self._check_matches_eager(model)

    def test_export_attention_ignores_the_configured_attention(self) -> None:
        # transformers keeps registered attention process-wide, so a wrapper
        # built later must not change what an earlier one exports.
        ids = torch.tensor([[3, 4, 5, 6]])

        def ops(wrapper: LastTokenLM) -> collections.Counter:
            with torch.no_grad():
                ep = torch.export.export(wrapper, (ids, torch.arange(4)), strict=True)
            return _ops_except_casts(ep)

        wrapper = LastTokenLM(_tiny(), 16).eval()
        before = ops(wrapper)
        eager = _tiny()
        eager.config._attn_implementation = "eager"
        self.assertEqual(ops(LastTokenLM(eager, 16).eval()), before)
        self.assertEqual(ops(wrapper), before)

    def test_masks_export_cannot_build_are_refused(self) -> None:
        # Gemma 3's bidirectional attention overlays the causal mask.
        wrapper = LastTokenLM(_tiny("gemma3", use_bidirectional_attention=True), 16)
        with torch.no_grad():
            with self.assertRaisesRegex(NotImplementedError, "attention mask"):
                wrapper.eval()(torch.tensor([[3, 4, 5, 6]]), torch.arange(4))
        config = _tiny().config
        with self.assertRaisesRegex(NotImplementedError, "argument new_argument"):
            _layer_mask(1, kv_length=4, q_length=4, config=config, new_argument=True)

    def test_mask_functions_export_cannot_reproduce_are_refused(self) -> None:
        config = _tiny().config
        ids = torch.tensor([[0, 0, 1, 1]])
        for name, mask_function, local_size in (
            (
                "packed sequences",
                and_masks(causal_mask_function, packed_sequence_mask_function(ids)),
                None,
            ),
            (
                "bidirectional blocks",
                or_masks(causal_mask_function, blockwise_overlay(ids)),
                None,
            ),
            ("bidirectional", bidirectional_mask_function, None),
            ("bidirectional window", sliding_window_bidirectional_mask_function(2), 2),
            ("another window", sliding_window_causal_mask_function(3), 2),
        ):
            with self.subTest(name), self.assertRaises(NotImplementedError):
                _layer_mask(
                    1,
                    kv_length=4,
                    q_length=4,
                    local_size=local_size,
                    config=config,
                    mask_function=mask_function,
                )

    def test_layer_masks_match_transformers_mask_functions(self) -> None:
        # A Llama config has no sliding_window: the mask function alone says
        # which local mask a layer takes.
        config = _tiny().config
        for name, mask_function, local_size in (
            ("causal", causal_mask_function, None),
            ("sliding", sliding_window_causal_mask_function(3), 3),
            (
                "chunked",
                chunked_causal_mask_function(3, torch.zeros(1, dtype=torch.long)),
                3,
            ),
        ):
            with self.subTest(name):
                want = sdpa_mask(
                    1,
                    q_length=3,
                    kv_length=9,
                    q_offset=5,
                    mask_function=mask_function,
                    local_size=local_size,
                    allow_is_causal_skip=False,
                )
                got = _layer_mask(
                    1,
                    kv_length=9,
                    q_length=3,
                    q_offset=5,
                    local_size=local_size,
                    config=config,
                    mask_function=mask_function,
                )
                self.assertTrue(torch.equal(got, want))

    def _check_matches_eager(self, model: PreTrainedModel) -> None:
        """The wrapper and its export match an uncached pass through the model's
        transformers eager attention, over a prefill and single-token decodes."""
        ids = torch.tensor(_IDS)
        prefill = 6
        calls = [(0, prefill)] + [(i, i + 1) for i in range(prefill, ids.shape[1])]
        seq = torch.export.Dim("seq", min=1, max=15)
        want = _eager_logits(model, ids)
        wrapper = LastTokenLM(model, 16).eval()
        with torch.no_grad():
            exported = torch.export.export(
                wrapper,
                (ids[:, :prefill], torch.arange(prefill)),
                dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                strict=True,
            ).module()
            for module in (wrapper, exported):
                for start, end in calls:
                    got = module(ids[:, start:end], torch.arange(start, end))
                    self.assertTrue(
                        torch.allclose(
                            got[:, -1], want[:, end - 1], atol=1e-5, rtol=1e-4
                        )
                    )

    def test_sliding_window_layers_match_eager_past_the_window(self) -> None:
        ids = torch.tensor(_IDS)
        for arch in ("gemma2", "gemma3"):
            with self.subTest(arch=arch):
                model = _random_gemma(
                    arch, sliding_window=3, attn_logit_softcapping=None
                )
                unwindowed = copy.deepcopy(model)
                unwindowed.config.sliding_window = 64
                self.assertFalse(
                    torch.allclose(
                        _eager_logits(unwindowed, ids),
                        _eager_logits(model, ids),
                        atol=1e-3,
                    )
                )
                self._check_matches_eager(model)

    def test_gemma4_layers_match_eager_past_the_window(self) -> None:
        for config_cls in (Gemma4TextConfig, Gemma4UnifiedTextConfig):
            with self.subTest(config_cls.__name__):
                self._check_matches_eager(_random_gemma4(config_cls))

    def test_gemma4_moe_layers_match_eager(self) -> None:
        self._check_matches_eager(_random_gemma4(Gemma4TextConfig, **_GEMMA4_MOE))

    def test_attention_logit_softcapping_matches_eager_when_configured_eager(
        self,
    ) -> None:
        # Eager's own mask is additive; the export takes a boolean one.
        model = _random_gemma("gemma2", sliding_window=3, attn_logit_softcapping=1.0)
        model.config._attn_implementation = "eager"
        self._check_matches_eager(model)

    def test_attention_logit_softcapping_matches_eager(self) -> None:
        ids = torch.tensor(_IDS)
        model = _random_gemma("gemma2", sliding_window=3, attn_logit_softcapping=1.0)
        uncapped = copy.deepcopy(model)
        for layer in uncapped.model.layers:
            layer.self_attn.attn_logit_softcapping = None
        self.assertFalse(
            torch.allclose(
                _eager_logits(uncapped, ids), _eager_logits(model, ids), atol=1e-3
            )
        )
        self._check_matches_eager(model)

    def test_attention_rejects_arguments_export_would_drop(self) -> None:
        config = GptOssConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=2,
            num_experts_per_tok=1,
        )
        model = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
        model.config._attn_implementation = "sdpa"
        model.generation_config.cache_implementation = "static"
        wrapper = LastTokenLM(model, 16).eval()
        ids = torch.tensor([[3, 4, 5, 6]])
        # GPT-OSS passes its attention sinks, which SDPA would drop.
        message = r"GptOssAttention \(layer 0\) passes s_aux=<tensor \(4,\)>"

        with torch.no_grad():
            with self.assertRaisesRegex(NotImplementedError, message):
                wrapper(ids, torch.arange(4))
            with self.assertRaisesRegex(Exception, message):
                torch.export.export(wrapper, (ids, torch.arange(4)), strict=True)

    def test_attention_checks_argument_values(self) -> None:
        module = torch.nn.Linear(1, 1)
        _check_attention_args(
            module,
            {
                "scaling": 0.25,
                "dropout": 0.0,
                "sliding_window": 4096,
                "use_cache": True,
                "position_ids": torch.arange(4),
                "cache_position": torch.arange(4),
                "cu_seq_lens_q": None,
            },
        )
        for name, value in (
            ("dropout", 0.1),
            ("cu_seq_lens_q", torch.tensor([0, 4])),
            ("head_mask", torch.ones(4)),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(NotImplementedError, f"passes {name}="):
                    _check_attention_args(module, {name: value})

    def test_attention_accepts_every_mapped_architecture(self) -> None:
        ids = torch.tensor([[3, 4, 5, 6]])
        for arch in _CONFIGS:
            with self.subTest(arch=arch):
                wrapper = LastTokenLM(_tiny(arch), 16).eval()
                with torch.no_grad():
                    logits = wrapper(ids, torch.arange(4))
                self.assertEqual(logits.shape, (1, 1, 128))

    def test_gguf_dequantized_twin_is_the_quantized_model_in_float(self) -> None:
        config = LlamaConfig(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=64,
            tie_word_embeddings=True,
            pad_token_id=0,
        )
        base = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
        base.config._attn_implementation = "sdpa"
        base.generation_config.cache_implementation = "static"
        pairs = _k_quant_pairs(base, torch.Generator().manual_seed(5))
        quantized, twin = copy.deepcopy(base), copy.deepcopy(base)
        load_gguf(quantized, "model.gguf", "llama", lambda _: iter(pairs))

        load_gguf(twin, "model.gguf", "llama", lambda _: iter(pairs), dequantize=True)

        self.assertFalse(
            any(isinstance(p, TorchAOBaseTensor) for p in twin.parameters())
        )
        self.assertEqual({p.dtype for p in twin.parameters()}, {torch.float32})
        self.assertIs(twin.lm_head.weight, twin.model.embed_tokens.weight)
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            self.assertTrue(torch.equal(twin(ids).logits, quantized(ids).logits))
        seq = torch.export.Dim("seq", min=1, max=15)
        graphs, consumers, embeddings = [], [], []
        for model in (quantized, twin):
            with torch.no_grad():
                ep = torch.export.export(
                    LastTokenLM(model, 16).eval(),
                    (ids, torch.arange(6)),
                    dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                    strict=True,
                )
            consumers.append(_constant_consumers(ep))
            # Decomposing unwraps the GGUF weights into dequantize_gguf nodes, as
            # lowering does.
            graph = ep.run_decompositions({}).graph
            graphs.append([n.target for n in graph.nodes if n.op == "call_function"])
            embeddings.append(
                [
                    n.args[2:]
                    for n in graph.nodes
                    if n.target == torch.ops.aten.embedding.default
                ]
            )
        # One constant feeds both the embedding and the tied lm_head, in both.
        self.assertEqual(consumers[0], consumers[1])
        self.assertEqual(
            consumers[1]["cached.model.model.embed_tokens.weight"],
            ["aten.embedding.default", "aten.linear.default"],
        )
        self.assertNotIn("cached.model.lm_head.weight", consumers[1])
        dequantize_gguf = torch.ops.torchao.dequantize_gguf.default
        # 14 decoder matrices, the embedding and its tied lm_head.
        self.assertEqual(graphs[0].count(dequantize_gguf), 16)
        self.assertNotIn(dequantize_gguf, graphs[1])
        self.assertEqual([t for t in graphs[0] if t != dequantize_gguf], graphs[1])
        # The same embedding literals, padding_idx included.
        self.assertEqual(embeddings[0], embeddings[1])

    def test_dequantize_gguf_flag_needs_gguf(self) -> None:
        argv = ["export_hf_llm", "--model", "unused", "--dequantize_gguf"]
        stderr = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(sys, "stderr", stderr),
        ):
            with mock.patch.object(export_hf_llm, "_load") as load:
                with self.assertRaises(SystemExit):
                    export_hf_llm.main()
        load.assert_not_called()
        self.assertIn("--dequantize_gguf needs --gguf", stderr.getvalue())

    def test_main_takes_the_package_before_the_lowering_report(self) -> None:
        argv = ["export_hf_llm", "--model", "config", "--output", "out.ptn"]
        calls = mock.Mock()
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(export_hf_llm, "_load"),
            mock.patch.object(export_hf_llm, "LastTokenLM") as wrapper,
            mock.patch.object(export_hf_llm, "export") as export,
            mock.patch.object(export_hf_llm, "to_native", calls.to_native),
            mock.patch.object(export_hf_llm, "report_lowering", calls.report_lowering),
        ):
            wrapper.return_value.eval.return_value.tokens_per_call = None
            export_hf_llm.main()
        ep = export.return_value
        self.assertEqual(
            calls.mock_calls,
            [
                mock.call.to_native(ep, metadata=None),
                mock.call.report_lowering(ep, "forward"),
                mock.call.to_native().save("out.ptn"),
            ],
        )

    def test_gguf_flags_reach_load_gguf(self) -> None:
        for dequantize in (False, True):
            argv = ["export_hf_llm", "--model", "config", "--gguf", "model.gguf"]
            argv += ["--dequantize_gguf"] if dequantize else []
            with (
                self.subTest(dequantize=dequantize),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(export_hf_llm, "_load") as load,
                mock.patch.object(
                    export_hf_llm, "load_gguf", side_effect=RuntimeError("loaded")
                ) as load_gguf,
            ):
                with self.assertRaisesRegex(RuntimeError, "loaded"):
                    export_hf_llm.main()
                load.assert_called_once_with(
                    "config", 0, False, empty_weights=True, offgraph_cache=False
                )
                load_gguf.assert_called_once_with(
                    load.return_value,
                    "model.gguf",
                    truncated=False,
                    dequantize=dequantize,
                )
