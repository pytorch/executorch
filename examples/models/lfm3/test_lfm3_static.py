# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared-checkpoint parity and the QNN prefill/decode cache contract."""

import unittest
from collections.abc import Sequence

import torch
from executorch.backends.qualcomm.quantizer.custom_annotation import annotate_conv_state
from executorch.backends.qualcomm.quantizer.rules import Q_ANNOTATION_KEY
from executorch.examples.models.lfm3.convert_weights import lfm_3_to_meta
from executorch.examples.models.lfm3.test_utils import (
    generic_model,
    random_hf_state_dict,
    run,
    tiny_hf_config,
    tiny_meta_config,
    VOCAB,
    WINDOW,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    ShortConvHybridLlamaModel,
)

CTX: int = 32


def static_model(ar_len: int) -> ShortConvHybridLlamaModel:
    config = tiny_meta_config()
    config.update(max_context_len=CTX, max_seq_len=CTX, use_kv_cache=True)
    model = ShortConvHybridLlamaModel(ModelArgs(**config), ar_len=ar_len).eval()
    model.load_state_dict(
        lfm_3_to_meta(random_hf_state_dict(), tiny_hf_config()), strict=True
    )
    return model


class HybridRunner:
    """KVManager smart mask and ConvStateManager valid-row updates."""

    def __init__(self, decode: ShortConvHybridLlamaModel):
        n_attn = len(decode.attention_layers)
        kvh, hd = decode.n_kv_heads, decode.head_dim
        self.k = [torch.zeros(1, kvh, hd, CTX - 1) for _ in range(n_attn)]
        self.v = [torch.zeros(1, kvh, CTX - 1, hd) for _ in range(n_attn)]
        # Stale rows must be invisible when a new sequence starts at position 0.
        generator = torch.Generator().manual_seed(2)
        self.conv = [
            torch.randn(1, decode.conv_state_len, decode.dim, generator=generator) * 100
            for _ in decode.conv_layers
        ]
        self.window = decode.sliding_window
        self.n_past = 0

    def step(
        self, model: ShortConvHybridLlamaModel, tokens: Sequence[int], n_valid: int
    ) -> torch.Tensor:
        ar = model.ar_len
        assert 0 < n_valid == len(tokens) <= ar
        cache_len = CTX - ar
        q_pos = self.n_past + torch.arange(ar)
        k_pos = torch.cat([torch.arange(cache_len), q_pos])
        valid = torch.cat(
            [torch.arange(cache_len) < self.n_past, torch.ones(ar, dtype=torch.bool)]
        )
        causal = valid[None] & (k_pos[None] <= q_pos[:, None])
        window = causal & (q_pos[:, None] - k_pos[None] < self.window)

        def mask(allowed: torch.Tensor) -> torch.Tensor:
            return torch.where(allowed, 0.0, -255.0)[None, None]

        toks = torch.zeros(1, ar, dtype=torch.int32)
        toks[0, :n_valid] = torch.tensor(tokens)
        with torch.no_grad():
            logits, k_out, v_conv = model(
                toks,
                mask(causal),
                mask(window),
                q_pos[None].int(),
                *[k[..., :cache_len] for k in self.k],
                *[v[:, :, :cache_len] for v in self.v],
                *self.conv,
            )
        n_attn = len(k_out)
        for index in range(n_attn if cache_len else 0):
            self.k[index][..., self.n_past : self.n_past + n_valid] = k_out[index][
                ..., :n_valid
            ]
            self.v[index][:, :, self.n_past : self.n_past + n_valid] = v_conv[index][
                :, :, :n_valid
            ]
        state_len = self.conv[0].shape[1]
        for index, new_rows in enumerate(v_conv[n_attn:]):
            self.conv[index] = torch.cat(
                [self.conv[index], new_rows[:, :n_valid]], dim=1
            )[:, -state_len:]
        self.n_past += n_valid
        return logits[0, :n_valid]


class Lfm3StaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = torch.randint(
            3, VOCAB, (20,), generator=torch.Generator().manual_seed(1)
        ).tolist()

    def one_shot(self) -> torch.Tensor:
        model = static_model(ar_len=CTX)
        return HybridRunner(model).step(model, self.tokens, len(self.tokens))

    def test_same_checkpoint_and_rope_tables(self) -> None:
        generic = generic_model(max_context_len=CTX)
        static = static_model(ar_len=1)
        generic_parameters = dict(generic.named_parameters())
        static_parameters = dict(static.named_parameters())
        self.assertEqual(set(generic_parameters), set(static_parameters))
        for key in generic_parameters:
            torch.testing.assert_close(
                generic_parameters[key], static_parameters[key], rtol=0, atol=0
            )
        for layer_type, prefix in (
            ("full_attention", ""),
            ("sliding_attention", "local_"),
        ):
            rope = generic.ropes[layer_type]
            for suffix in ("cos", "sin"):
                generic_freqs = getattr(rope, f"freqs_{suffix}")
                static_freqs = getattr(static, f"{prefix}freqs_{suffix}")
                torch.testing.assert_close(
                    generic_freqs[:, : generic_freqs.shape[-1] // 2],
                    static_freqs,
                    rtol=0,
                    atol=0,
                )

    def test_static_matches_generic(self) -> None:
        ref = run(generic_model(max_context_len=CTX), self.tokens, WINDOW)
        torch.testing.assert_close(self.one_shot(), ref, rtol=0, atol=1e-3)

    def test_prepared_static_matches_generic(self) -> None:
        model = static_model(ar_len=CTX)
        for layer in model.layers:
            if hasattr(layer, "attention"):
                layer.attention.prepare_attention_conv()
            layer.feed_forward.prepare_feedforward_conv()
        model.prepare_output_conv()
        got = HybridRunner(model).step(model, self.tokens, len(self.tokens))
        ref = run(generic_model(max_context_len=CTX), self.tokens, WINDOW)
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-3)

    def test_hybrid_runner_matches_one_shot(self) -> None:
        ref = self.one_shot()
        prefill, decode = static_model(ar_len=WINDOW), static_model(ar_len=1)
        runner = HybridRunner(decode)
        outputs = []
        prompt, rest = self.tokens[:13], self.tokens[13:]
        for pos in range(0, len(prompt), prefill.ar_len):
            chunk = prompt[pos : pos + prefill.ar_len]
            outputs.append(runner.step(prefill, chunk, len(chunk)))
        for token in rest:
            outputs.append(runner.step(decode, [token], 1))
        torch.testing.assert_close(torch.cat(outputs), ref, rtol=0, atol=1e-3)

    def test_reused_runner_resets_at_position_zero(self) -> None:
        model = static_model(ar_len=1)
        runner = HybridRunner(model)
        for token in self.tokens:
            runner.step(model, [token], 1)
        runner.n_past = 0
        got = torch.cat([runner.step(model, [token], 1) for token in self.tokens])
        torch.testing.assert_close(got, self.one_shot(), rtol=0, atol=1e-3)

    def test_conv_state_annotation_skips_kv_cache(self) -> None:
        model = static_model(ar_len=4)
        runner = HybridRunner(model)
        cache_len = CTX - model.ar_len
        mask = torch.zeros(1, 1, model.ar_len, CTX)
        inputs = (
            torch.zeros(1, model.ar_len, dtype=torch.int32),
            mask,
            mask,
            torch.arange(model.ar_len, dtype=torch.int32)[None],
            *[k[..., :cache_len] for k in runner.k],
            *[v[:, :, :cache_len] for v in runner.v],
            *runner.conv,
        )
        gm = torch.export.export(model, inputs, strict=False).module()
        annotate_conv_state(gm)
        annotated = [
            node
            for node in gm.graph.nodes
            if node.target
            in (torch.ops.aten.cat.default, torch.ops.aten.concat.default)
            and Q_ANNOTATION_KEY in node.meta
        ]
        self.assertEqual(len(annotated), len(model.conv_layers))
        for node in annotated:
            self.assertEqual(node.args[0][0].meta["val"].dim(), 3)
