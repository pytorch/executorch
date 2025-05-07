# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import precompute_freqs_cis

from executorch.backends.qualcomm.utils.utils import TMANLinear


def apply_rotary_emb_single(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    # The implementation of RoPE in HuggingFace processes query and key with two half instead of interleaved way.
    # The main difference is stride in StrideSlice op. For interleaved way, stride is two which is not friendly for HTP backend.
    # Ref: https://github.com/huggingface/transformers/issues/25199
    x_r, x_i = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # broadcast for batch_prefill mode input x
    if x.dim() == 4:
        freqs_cos = freqs_cos[None, None, :, :]
        freqs_sin = freqs_sin[None, None, :, :]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_out = torch.cat([x_out_r, x_out_i], dim=-1)
    return x_out


def permute(qlinear: nn.Module, n_head: int, n_head_kv: int | None):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    # Unpack the qzeros first for permutations
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qlinear.qzeros, 2).expand(-1, -1, qlinear.pack_factor),
        qlinear.wf_unsqueeze_zero,
    ).to(qlinear.dequant_dtype)
    zeros = torch.bitwise_and(zeros, qlinear.maxq).reshape(qlinear.scales.shape)

    def _permute(weights: torch.Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(weights.shape[0], n_head, 2, weights.shape[1] // n_head // 2)
                .swapaxes(-2, -1)
                .reshape(weights.shape))

    qweight = _permute(qlinear.qweight, n_head, n_head_kv)
    scales = _permute(qlinear.scales, n_head, n_head_kv)
    zeros = _permute(zeros, n_head, n_head_kv)

    zeros = zeros.numpy().astype(qlinear.pack_np_math_dtype)
    qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // qlinear.pack_dtype_bits * qlinear.bits), dtype=qlinear.pack_np_math_dtype)
    for col in range(qzeros.shape[1]):
        for j in range(qlinear.pack_factor):
            qzeros[:, col] |= zeros[:, col * qlinear.pack_factor + j] << (qlinear.bits * j)
    qzeros = torch.from_numpy(qzeros.astype(qlinear.pack_np_dtype))

    return qweight, scales, qzeros


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = config.n_heads // self.n_kv_heads
        self.max_seq_len = config.max_seq_len
        self.output_new_cache_only = output_new_cache_only

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        self.scale = float(self.head_dim) ** 0.5

    def prepare_sha(self):
        self.wq_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=False)
                for _ in range(self.n_heads)
            ]
        )
        self.wk_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=False)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wv_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=False)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wo_sha = nn.Conv2d(self.n_heads * self.head_dim, self.dim, 1, bias=False)

        self.forward_mha = self.forward
        self.forward = self.forward_sha
        for i in range(self.n_heads):
            self.wq_sha[i].weight.data.copy_(
                self.wq.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
        for i in range(self.n_kv_heads):
            self.wk_sha[i].weight.data.copy_(
                self.wk.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            self.wv_sha[i].weight.data.copy_(
                self.wv.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
        self.wo_sha.weight.data.copy_(self.wo.weight[:, :, None, None])

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: Optional[List[torch.Tensor]] = None,
        v_caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        # In the HTP backend, the input axis order for the convolution operation is
        # more efficient with [1, 1, seq_len, dim] compared to [1, seq_len, 1, dim].
        hidden_states = torch.reshape(
            hidden_states, (bsz, seq_len, 1, self.dim)
        ).transpose(1, 3)
        q = [
            wq_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wq_sha in self.wq_sha
        ]
        k = [
            wk_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wk_sha in self.wk_sha
        ]
        v = [
            wv_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wv_sha in self.wv_sha
        ]
        for i in range(len(q)):
            q[i] = apply_rotary_emb_single(q[i], freqs_cos, freqs_sin)
        for i in range(len(k)):
            k[i] = apply_rotary_emb_single(k[i], freqs_cos, freqs_sin).transpose(1, 2)

        output_y = []
        kh, vh = [], []
        # kv cache mode
        if k_caches and v_caches:
            for i, _ in enumerate(k_caches):
                kh.append(torch.cat([k_caches[i], k[i]], dim=-1))
                vh.append(torch.cat([v_caches[i], v[i]], dim=1))
        # batch_prefill mode
        else:
            kh = k
            vh = v

        for i, _ in enumerate(q):
            cache_idx = i // self.num_key_value_groups
            attn = q[i] @ kh[cache_idx]
            attn = attn / self.scale + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ vh[cache_idx]

            output_y.append(y)

        y = torch.concat(output_y, dim=-1)
        y = y.reshape(bsz, seq_len, 1, -1)
        y = y.transpose(1, 3)
        y = self.wo_sha(y)
        y = y.transpose(1, 3)
        y = y.reshape(bsz, seq_len, -1)

        if self.output_new_cache_only:
            return y, k, v

        return y, kh, vh

    # TODO: can't find a better way to replace with tman_sha at this moment
    #       place it here for now
    def prepare_tman(self, do_permute=False, use_sha=False):
        from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
        assert(isinstance(self.wq, TorchQuantLinear))
        assert(isinstance(self.wk, TorchQuantLinear))
        assert(isinstance(self.wv, TorchQuantLinear))
        assert(isinstance(self.wo, TorchQuantLinear))
        self.wq.post_init()
        self.wk.post_init()
        self.wv.post_init()
        self.wo.post_init()

        if use_sha:
            self.wq_tman = nn.ModuleList(
                [
                    TMANLinear(self.wq, n_splits=self.n_heads)
                    for _ in range(self.n_heads)
                ]
            )
            self.wk_tman = nn.ModuleList(
                [
                    TMANLinear(self.wk, n_splits=self.n_kv_heads)
                    for _ in range(self.n_kv_heads)
                ]
            )
            self.wv_tman = nn.ModuleList(
                [
                    TMANLinear(self.wv, n_splits=self.n_kv_heads)
                    for _ in range(self.n_kv_heads)
                ]
            )
        else:
            self.wq_tman = TMANLinear(self.wq)
            self.wk_tman = TMANLinear(self.wk)
            self.wv_tman = TMANLinear(self.wv)
        self.wo_tman = TMANLinear(self.wo)

        self.forward_mha = self.forward
        self.forward = self.forward_tman

        if do_permute:
            wq_qweight, wq_scales, wq_qzeros = permute(self.wq, self.n_heads, self.n_heads)
            wk_qweight, wk_scales, wk_qzeros = permute(self.wk, self.n_heads, self.n_kv_heads)
        else:
            wq_qweight, wq_scales, wq_qzeros = self.wq.qweight, self.wq.scales, self.wq.qzeros
            wk_qweight, wk_scales, wk_qzeros = self.wk.qweight, self.wk.scales, self.wk.qzeros
        wv_qweight, wv_scales, wv_qzeros = self.wv.qweight, self.wv.scales, self.wv.qzeros

        def _copy(target: nn.ModuleList | TMANLinear, source: TorchQuantLinear, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, n_head: int, n_head_kv: int | None):
            if use_sha:
                if n_head_kv is not None and n_head != n_head_kv:
                    n_head = n_head_kv
                for i in range(n_head):
                    target[i].qweight.copy_(
                        qweight[
                            :, i * self.head_dim : (i + 1) * self.head_dim
                        ]
                    )
                    target[i].scales.copy_(
                        scales[
                            :, i * self.head_dim : (i + 1) * self.head_dim
                        ]
                    )
                    target[i].qzeros.copy_(
                        qzeros[
                            :, i * self.head_dim // source.pack_factor : (i + 1) * self.head_dim // source.pack_factor
                        ]
                    )
            else:
                target.qweight.copy_(qweight)
                target.scales.copy_(scales)
                target.qzeros.copy_(qzeros)

        _copy(self.wq_tman, self.wq, wq_qweight, wq_scales, wq_qzeros, self.n_heads, self.n_heads)
        _copy(self.wk_tman, self.wk, wk_qweight, wk_scales, wk_qzeros, self.n_heads, self.n_kv_heads)
        _copy(self.wv_tman, self.wv, wv_qweight, wv_scales, wv_qzeros, self.n_heads, self.n_kv_heads)

    def forward_tman(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: Optional[List[torch.Tensor]] = None,
        v_caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape

        if isinstance(self.wq_tman, nn.ModuleList):
            q = [
                wq_tman(hidden_states)
                for wq_tman in self.wq_tman
            ]
            k = [
                wk_tman(hidden_states)
                for wk_tman in self.wk_tman
            ]
            v = [
                wv_tman(hidden_states)
                for wv_tman in self.wv_tman
            ]
            for i in range(len(q)):
                q[i] = apply_rotary_emb_single(q[i], freqs_cos, freqs_sin)
            for i in range(len(k)):
                k[i] = apply_rotary_emb_single(k[i], freqs_cos, freqs_sin).permute(0, 2, 1)
        else:
            q = self.wq_tman(hidden_states).reshape(bsz, seq_len, self.n_heads, self.head_dim)
            k = self.wk_tman(hidden_states).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
            v = self.wv_tman(hidden_states).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
            q = apply_rotary_emb_single(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb_single(k, freqs_cos, freqs_sin).permute(0, 2, 3, 1)
            # Use split_with_sizes as only split_with_sizes is supported
            q = torch.split_with_sizes(q, [1 for _ in range(self.n_heads)], dim=2)
            k = torch.split_with_sizes(k, [1 for _ in range(self.n_kv_heads)], dim=1)
            v = torch.split_with_sizes(v, [1 for _ in range(self.n_kv_heads)], dim=2)
            q = [t.squeeze(2) for t in q]
            k = [t.squeeze(1) for t in k]
            v = [t.squeeze(2) for t in v]

        output_y = []
        kh, vh = [], []
        # kv cache mode
        if k_caches and v_caches:
            for i, _ in enumerate(k_caches):
                kh.append(torch.cat([k_caches[i], k[i]], dim=-1))
                vh.append(torch.cat([v_caches[i], v[i]], dim=1))
        # batch_prefill mode
        else:
            kh = k
            vh = v

        for i, _ in enumerate(q):
            cache_idx = i // self.num_key_value_groups
            attn = q[i] @ kh[cache_idx]
            attn = attn / self.scale + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ vh[cache_idx]

            output_y.append(y)

        y = torch.concat(output_y, dim=-1)
        y = self.wo_tman(y)

        if self.output_new_cache_only:
            return y, k, v

        return y, kh, vh

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape

        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb_single(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb_single(k, freqs_cos, freqs_sin).permute(0, 2, 3, 1)

        output_kh, output_vh, output_y = [], [], []
        kh, vh = [], []
        # kv cache mode
        if k_caches and v_caches:
            for i, _ in enumerate(k_caches):
                kh.append(torch.cat([k_caches[i], k[:, i, :, :]], dim=-1))
                vh.append(torch.cat([v_caches[i], v[:, :, i, :]], dim=1))
            for i in range(self.n_heads):
                cache_idx = i // self.num_key_value_groups

                attn = q[:, :, i, :] @ kh[cache_idx]
                attn = attn / self.scale + atten_mask
                attn = self.attn_softmax(attn)
                y = attn @ vh[cache_idx]

                output_y.append(y)

        # batch_prefill mode
        else:
            kh = k
            vh = v
            for i in range(self.n_heads):
                cache_idx = i // self.num_key_value_groups

                attn = q[:, :, i, :] @ kh[:, cache_idx, :, :]
                attn = attn / self.scale + atten_mask
                attn = self.attn_softmax(attn)
                y = attn @ vh[:, :, cache_idx, :]

                output_y.append(y)

        for i in range(self.n_kv_heads):
            if self.output_new_cache_only:
                output_kh.append(k[:, i, :, -1])
                output_vh.append(v[:, -1, i, :])
            else:
                output_kh.append(k[:, i, :, :])
                output_vh.append(v[:, :, i, :])

        y = torch.concat(output_y, dim=-1)
        y = self.wo(y)

        return y, output_kh, output_vh


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.hidden_dim is not None
        self.hidden_dim: int = args.hidden_dim
        self.dim: int = args.dim
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def prepare_feedfoward_conv(self):
        self.w1_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)
        self.w2_conv = nn.Conv2d(self.hidden_dim, self.dim, 1, bias=False)
        self.w3_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_feedfoward_conv

        self.w1_conv.weight.data.copy_(self.w1.weight[:, :, None, None])
        self.w2_conv.weight.data.copy_(self.w2.weight[:, :, None, None])
        self.w3_conv.weight.data.copy_(self.w3.weight[:, :, None, None])

        del self.w1
        del self.w2
        del self.w3

    def forward_feedfoward_conv(self, x):
        bsz, _, _ = x.size()
        x = torch.reshape(x, (bsz, -1, 1, self.dim))
        x = x.transpose(1, 3)  # Transpose right before and after Conv
        x = self.w2_conv(F.silu(self.w1_conv(x)) * self.w3_conv(x))
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, -1, self.dim))
        return x

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = config.dim
        self.attention = LlamaAttention(
            config=config, output_new_cache_only=output_new_cache_only
        )
        self.feed_forward = FeedForward(config)
        self.attention_norm = torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = torch.nn.RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k_cache, v_cache = self.attention(
            hidden_states=self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            atten_mask=atten_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        h = x + h
        output = h + self.feed_forward(self.ffn_norm(h))
        return output, k_cache, v_cache


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
    ):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.rope_freq_base = config.rope_freq_base
        self.use_kv_cache = config.use_kv_cache
        self.ar_len = ar_len
        self.output_new_cache_only = output_new_cache_only
        self.use_i64_token = use_i64_token
        self.output_cache = output_cache

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, self.output_new_cache_only)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len,
            config.rope_freq_base,
            config.use_scaled_rope,
            config.rope_scale_factor,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def prepare_output_conv(self):
        def forward_output_conv(x):
            bsz, _, _ = x.size()
            x = torch.reshape(x, (bsz, -1, 1, self.dim))
            x = x.transpose(1, 3)  # Transpose right before and after Conv
            x = self.output_conv(x)
            x = x.transpose(1, 3)
            x = torch.reshape(x, (bsz, -1, self.vocab_size))
            return x

        self.output_conv = nn.Conv2d(self.dim, self.vocab_size, 1, bias=False)
        self.output_conv.weight.data.copy_(self.output.weight[:, :, None, None])

        del self.output
        self.output = forward_output_conv

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        hidden_states = self.tok_embeddings(tokens)
        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind * self.n_kv_heads
                offset_v = self.n_layers * self.n_kv_heads + offset_k
                k_caches = args[offset_k : offset_k + self.n_kv_heads]
                v_caches = args[offset_v : offset_v + self.n_kv_heads]
            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self, use_kv_cache=True):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )

        atten_mask = torch.full((self.ar_len, self.ar_len), torch.tensor(-255.0))
        mask_cond = torch.arange(atten_mask.size(-1))
        atten_mask.masked_fill_(
            mask_cond < (mask_cond + 1).view(atten_mask.size(-1), 1), 0
        )
        if self.max_seq_len != self.ar_len:
            atten_mask = torch.cat(
                [
                    torch.ones(self.ar_len, self.max_seq_len - self.ar_len) * -255.0,
                    atten_mask,
                ],
                dim=-1,
            )
        atten_mask = atten_mask[None, :, :].expand(
            self.max_batch_size, self.ar_len, self.max_seq_len
        )
        if use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []

            for _ in range(self.n_layers):
                for _ in range(self.n_kv_heads):
                    # transpose first to decrease the runtime efforts
                    k_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.head_dim,
                            self.max_seq_len - self.ar_len,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.max_seq_len - self.ar_len,
                            self.head_dim,
                        )
                    )
            return (
                tokens,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            tokens,
            atten_mask,
        )

    def get_metadata(self):
        # TODO: modify this when enabling LLAMA 7B
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_dim": self.dim,
            "get_head_dim": self.head_dim,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_kv_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
        }
