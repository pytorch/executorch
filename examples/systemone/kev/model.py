# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, l2norm


class Backbone(nn.Module):
    """Qwen3.5 prefill with explicit convolution, DeltaNet, and attention state."""

    def __init__(self, lm, backend):
        super().__init__()
        if lm.config.model_type != "qwen3_5_text":
            raise ValueError("This example supports Kev's dense Qwen3.5 backbone")
        if lm.config.rope_parameters["rope_type"] != "default":
            raise ValueError("Expected default RoPE")
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.register_buffer("inv_freq", lm.rotary_emb.inv_freq.float().clone())
        self.backend = backend
        if backend == "xnnpack":
            from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
        elif backend == "mlx":
            import executorch.backends.mlx.custom_kernel_ops.gated_delta_rule  # noqa: F401
            import executorch.backends.mlx.custom_ops  # noqa: F401
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _linear_attention(self, attn, x, conv, recurrent):
        batch, length, _ = x.shape
        qkv = attn.in_proj_qkv(x).transpose(1, 2)
        if conv is None:
            conv = qkv.new_zeros(batch, attn.conv_dim, attn.conv_kernel_size)
            recurrent = torch.zeros(
                batch,
                attn.num_v_heads,
                attn.head_k_dim,
                attn.head_v_dim,
                device=x.device,
                dtype=torch.float32,
            )
        history = torch.cat((conv.expand(batch, -1, -1), qkv), dim=-1)
        conv_out = history[:, :, -attn.conv_kernel_size :].contiguous()
        qkv = F.silu(
            F.conv1d(history, attn.conv1d.weight, groups=attn.conv_dim)[:, :, -length:]
        ).transpose(1, 2)
        q, k, v = qkv.split((attn.key_dim, attn.key_dim, attn.value_dim), dim=-1)
        q = l2norm(q.reshape(batch, length, -1, attn.head_k_dim).float())
        k = l2norm(k.reshape(batch, length, -1, attn.head_k_dim).float())
        v = v.reshape(batch, length, -1, attn.head_v_dim).float()
        q = q * attn.head_k_dim**-0.5
        repeats = attn.num_v_heads // attn.num_k_heads
        if repeats > 1:
            q = q.repeat_interleave(repeats, dim=2)
            k = k.repeat_interleave(repeats, dim=2)
        beta = attn.in_proj_b(x).sigmoid().float()
        g = -attn.A_log.float().exp() * F.softplus(
            attn.in_proj_a(x).float() + attn.dt_bias
        )
        decay = g.exp()
        recurrent = recurrent.expand(batch, -1, -1, -1)
        if self.backend == "mlx":
            # MLX uses [B, H, V, K] and mutates its input. A prefix stays immutable.
            state = recurrent.transpose(-1, -2).clone(
                memory_format=torch.contiguous_format
            )
            y = torch.ops.mlx.gated_delta_rule(q, k, v, decay, beta, state)
            recurrent_out = state.transpose(-1, -2).contiguous()
        else:
            y, recurrent_out = torch.ops.llama.gated_delta_rule(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                decay.transpose(1, 2).contiguous(),
                beta.transpose(1, 2).contiguous(),
                recurrent.contiguous(),
            )
            y = y.transpose(1, 2)
        z = attn.in_proj_z(x).reshape(-1, attn.head_v_dim)
        y = attn.norm(y.to(x.dtype).reshape(-1, attn.head_v_dim), z)
        return attn.out_proj(y.reshape(batch, length, -1)), conv_out, recurrent_out

    def _full_attention(self, attn, x, cos, sin, kv):
        batch, length, _ = x.shape
        q, gate = (
            attn.q_proj(x)
            .reshape(batch, length, -1, 2 * attn.head_dim)
            .chunk(2, dim=-1)
        )
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(
            attn.k_proj(x).reshape(batch, length, -1, attn.head_dim)
        ).transpose(1, 2)
        v = attn.v_proj(x).reshape(batch, length, -1, attn.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        start = 0
        if kv is not None:
            start = kv.shape[-2]
            k = torch.cat((kv[0].expand(batch, -1, -1, -1), k), dim=2)
            v = torch.cat((kv[1].expand(batch, -1, -1, -1), v), dim=2)
        if self.backend == "mlx":
            y = torch.ops.mlx.custom_sdpa(
                q, k, v, start_pos=start, is_causal=True, scale=attn.scaling
            )
        else:
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= (
                torch.arange(length, device=x.device)[:, None] + start
            )
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=attn.scaling, enable_gqa=True
            )
        y = y.transpose(1, 2).reshape(batch, length, -1)
        y = y * gate.reshape(batch, length, -1).sigmoid()
        return attn.o_proj(y), torch.stack((k, v))

    def forward(self, tokens, conv=None, recurrent=None, kv=None):
        x = self.embed_tokens(tokens)
        start = 0 if kv is None else kv.shape[-2]
        positions = torch.arange(tokens.shape[1], device=tokens.device) + start
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos, sin = freqs.cos()[None].to(x.dtype), freqs.sin()[None].to(x.dtype)
        conv_out, recurrent_out, kv_out = [], [], []
        linear_index, full_index = 0, 0
        for layer in self.layers:
            h = layer.input_layernorm(x)
            if layer.block_type == "linear_attention":
                h, c, r = self._linear_attention(
                    layer.linear_attn,
                    h,
                    None if conv is None else conv[linear_index],
                    None if recurrent is None else recurrent[linear_index],
                )
                conv_out.append(c)
                recurrent_out.append(r)
                linear_index += 1
            else:
                h, state = self._full_attention(
                    layer.self_attn,
                    h,
                    cos,
                    sin,
                    None if kv is None else kv[full_index],
                )
                kv_out.append(state)
                full_index += 1
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return (
            self.norm(x),
            torch.stack(conv_out),
            torch.stack(recurrent_out),
            torch.stack(kv_out),
        )


class Prefill(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, tokens):
        return self.backbone(tokens)[1:]


class Score(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, tokens, decide, options, conv, recurrent, kv):
        h = self.backbone(tokens, conv, recurrent, kv)[0]
        batch, length, dim = h.shape
        offsets = torch.arange(batch, device=tokens.device) * length
        h = h.reshape(-1, dim)
        h_decide = h.index_select(0, decide + offsets).float()[:, None]
        h_options = (
            h.index_select(0, (options + offsets[:, None]).reshape(-1))
            .float()
            .reshape(batch, options.shape[1], dim)
        )
        return (
            (self.head.k(h_options) * self.head.q(h_decide)).sum(-1)
            * self.head.scale
            / self.head.temperature
        )
