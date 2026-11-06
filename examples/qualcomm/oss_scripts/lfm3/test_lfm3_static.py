# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerics tests for the QNN static LFM3 model.

Hybrid runner contract in Python: AR-N prefill chunks + AR-1 decode with
smart-mask KV caches and ConvStateManager-style conv state, vs one shot.
"""

import unittest

import torch
from executorch.examples.qualcomm.oss_scripts.lfm3.convert_weights import (
    lfm3_hf_to_static,
)
from executorch.examples.qualcomm.oss_scripts.lfm3.model_wrapper import (
    Lfm3Config,
    Lfm3Model,
)

CTX = 32
LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"]
DIM, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, VOCAB = 64, 128, 4, 2, 16, 97


def tiny_hf_config():
    """Tiny LFM3 config in HF terms (`rope_parameters` as in the HF checkpoint)."""
    return {
        "hidden_size": DIM,
        "num_hidden_layers": len(LAYER_TYPES),
        "intermediate_size": HIDDEN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "norm_eps": 1e-5,
        "conv_L_cache": 7,
        "sliding_window": 8,
        "layer_types": LAYER_TYPES,
        "rope_parameters": {
            "full_attention": {"rope_theta": 1e6, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_theta": 1e4},
        },
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def random_hf_state_dict(seed: int = 0):
    """Random weights with LFM3's HF parameter names and shapes."""
    g = torch.Generator().manual_seed(seed)

    def w(*shape, scale=0.1):
        return torch.randn(*shape, generator=g) * scale

    def norm(n):
        return 1.0 + w(n)

    sd = {
        "model.embed_tokens.weight": w(VOCAB, DIM, scale=1.0),
        "model.embedding_norm.weight": norm(DIM),
    }
    for i, t in enumerate(LAYER_TYPES):
        p = f"model.layers.{i}."
        for n in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"):
            sd[p + n + ".weight"] = norm(DIM)
        sd[p + "feed_forward.w1.weight"] = w(HIDDEN, DIM)
        sd[p + "feed_forward.w3.weight"] = w(HIDDEN, DIM)
        sd[p + "feed_forward.w2.weight"] = w(DIM, HIDDEN)
        if t == "conv":
            sd[p + "conv.in_proj.weight"] = w(3 * DIM, DIM)
            sd[p + "conv.out_proj.weight"] = w(DIM, DIM)
            sd[p + "conv.conv.weight"] = w(DIM, 1, 7, scale=0.3)
        else:
            a = p + "self_attn."
            sd[a + "q_proj.weight"] = w(HEADS * HEAD_DIM, DIM)
            sd[a + "k_proj.weight"] = w(KV_HEADS * HEAD_DIM, DIM)
            sd[a + "v_proj.weight"] = w(KV_HEADS * HEAD_DIM, DIM)
            sd[a + "o_proj.weight"] = w(DIM, HEADS * HEAD_DIM)
            sd[a + "gate_proj.weight"] = w(HEADS, DIM)
            sd[a + "q_norm.weight"] = norm(HEAD_DIM)
            sd[a + "k_norm.weight"] = norm(HEAD_DIM)
    return sd


def static_model(ar_len: int) -> Lfm3Model:
    cfg = Lfm3Config(**tiny_hf_config(), max_context_len=CTX, max_seq_len=CTX)
    model = Lfm3Model(cfg, ar_len=ar_len).eval()
    model.load_state_dict(lfm3_hf_to_static(random_hf_state_dict()), strict=True)
    return model


class HybridRunner:
    """Mirrors the C++ runner: KVManager smart mask + ConvStateManager.update."""

    def __init__(self, decode: Lfm3Model):
        n_attn = len(decode.attention_layers)
        kvh, hd = decode.n_kv_heads, decode.head_dim
        self.k = [torch.zeros(1, kvh, hd, CTX - 1) for _ in range(n_attn)]
        self.v = [torch.zeros(1, kvh, CTX - 1, hd) for _ in range(n_attn)]
        # Garbage on purpose: the graph must mask state rows before position 0.
        self.conv = [
            torch.randn(1, decode.conv_state_len, decode.config.hidden_size) * 100
            for _ in decode.conv_layers
        ]
        self.window = decode.config.sliding_window
        self.n_past = 0

    def step(self, model: Lfm3Model, tokens, n_valid: int):
        ar = model.ar_len
        cache_len = CTX - ar
        q_pos = self.n_past + torch.arange(ar)
        k_pos = torch.cat([torch.arange(cache_len), q_pos])
        valid = torch.cat(
            [torch.arange(cache_len) < self.n_past, torch.ones(ar, dtype=torch.bool)]
        )
        causal = valid[None] & (k_pos[None] <= q_pos[:, None])
        window = causal & (q_pos[:, None] - k_pos[None] < self.window)

        def mask(b):
            return torch.where(b, 0.0, -255.0)[None, None]

        toks = torch.zeros(1, ar, dtype=torch.int32)
        toks[0, : len(tokens)] = torch.tensor(tokens)
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
        for i in range(n_attn if cache_len else 0):
            self.k[i][..., self.n_past : self.n_past + n_valid] = k_out[i][
                ..., :n_valid
            ]
            self.v[i][:, :, self.n_past : self.n_past + n_valid] = v_conv[i][
                :, :, :n_valid
            ]
        state_len = self.conv[0].shape[1]
        for i, new_rows in enumerate(v_conv[n_attn:]):
            self.conv[i] = torch.cat([self.conv[i], new_rows[:, :n_valid]], dim=1)[
                :, -state_len:
            ]
        self.n_past += n_valid
        return logits[0, :n_valid]


class Lfm3StaticTest(unittest.TestCase):
    tokens = torch.randint(
        3, 97, (20,), generator=torch.Generator().manual_seed(1)
    ).tolist()

    def one_shot(self):
        model = static_model(ar_len=CTX)
        runner = HybridRunner(model)
        runner.k = [
            torch.zeros(1, model.n_kv_heads, model.head_dim, 0) for _ in runner.k
        ]
        runner.v = [
            torch.zeros(1, model.n_kv_heads, 0, model.head_dim) for _ in runner.v
        ]
        runner.conv = [torch.zeros_like(c) for c in runner.conv]
        with torch.no_grad():
            return runner.step(model, self.tokens, len(self.tokens))

    def test_hybrid_runner_matches_one_shot(self):
        ref = self.one_shot()
        prefill, decode = static_model(ar_len=8), static_model(ar_len=1)
        runner = HybridRunner(decode)
        out = []
        prompt, rest = (
            self.tokens[:13],
            self.tokens[13:],
        )  # 13 = full chunk + partial chunk
        for s in range(0, len(prompt), prefill.ar_len):
            chunk = prompt[s : s + prefill.ar_len]
            out.append(runner.step(prefill, chunk, len(chunk)))
        for t in rest:
            out.append(runner.step(decode, [t], 1))
        got = torch.cat(out)
        self.assertLess((got - ref).abs().max().item(), 1e-3)

    def test_conv_state_annotation_skips_kv_cache(self):
        # KV updates are also cat(placeholder, graph output); they must keep
        # annotate_kv_8bit's encoding.
        from executorch.backends.qualcomm.quantizer.custom_annotation import (
            annotate_conv_state,
        )
        from executorch.backends.qualcomm.quantizer.rules import Q_ANNOTATION_KEY

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
            n
            for n in gm.graph.nodes
            if n.target in (torch.ops.aten.cat.default, torch.ops.aten.concat.default)
            and Q_ANNOTATION_KEY in n.meta
        ]
        self.assertEqual(len(annotated), len(model.conv_layers))
        for n in annotated:
            self.assertEqual(n.args[0][0].meta["val"].dim(), 3)


if __name__ == "__main__":
    unittest.main()
