# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Straightforward eager Phonon for tests: interleaved RoPE on the original
in_proj layout, a growing KV cache, and Python-side streaming state, written
to follow the official runtime operation by operation. The export programs in
model.py must compute the same function."""

import math

import torch
import torch.nn.functional as F

from executorch.examples.models.phonon.model import Phonon, rope_inv_freq


def rope_interleaved(x, pos, max_period):
    """x [B, T, H, hd], absolute positions pos [T]; pairs (2i, 2i+1) rotate."""
    inv = rope_inv_freq(x.shape[-1], max_period)
    ang = pos.to(torch.float32)[:, None] * inv[None, :]
    c, s = torch.cos(ang)[None, :, None, :], torch.sin(ang)[None, :, None, :]
    xr, xi = x[..., 0::2], x[..., 1::2]
    return torch.stack([xr * c - xi * s, xr * s + xi * c], dim=-1).flatten(-2)


class FlowLMReference:
    def __init__(self, model: Phonon):
        assert model.rope_layout == "interleaved"
        self.m, self.cfg = model.flow_lm, model.cfg
        with torch.no_grad():
            self.cond = self.m.summed_conditions()
            self.t_cond = self.m.flow_net.lsd_time_condition()
        self.k = [None] * self.cfg.num_layers
        self.v = [None] * self.cfg.num_layers
        self.pos = 0

    @torch.no_grad()
    def backbone(self, x):
        cfg = self.cfg
        _, t, d = x.shape
        h, hd = cfg.num_heads, cfg.head_dim
        pos = torch.arange(self.pos, self.pos + t)
        for i, layer in enumerate(self.m.transformer.layers):
            qkv = layer.self_attn.in_proj(layer.norm1(x)).view(1, t, 3, h, hd)
            q = rope_interleaved(qkv[:, :, 0], pos, cfg.max_period)
            k = rope_interleaved(qkv[:, :, 1], pos, cfg.max_period)
            v = qkv[:, :, 2]
            self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k], 1)
            self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v], 1)
            kk, vv, qq = (z.transpose(1, 2) for z in (self.k[i], self.v[i], q))
            n = kk.shape[2]
            qpos = torch.arange(n - t, n)[:, None]
            mask = torch.where(torch.arange(n)[None, :] <= qpos, 0.0, float("-inf"))
            att = torch.softmax(qq @ kk.transpose(-1, -2) / math.sqrt(hd) + mask, -1)
            o = (att @ vv).transpose(1, 2).reshape(1, t, d)
            x = x + layer.self_attn.out_proj(o)
            x = x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))
        self.pos += t
        return x

    def prompt(self, embeds):
        self.backbone(embeds)

    @torch.no_grad()
    def step(self, prev_latent, noise):
        """prev_latent None (BOS) or [1, 1, ldim]; returns latent, eos_logit."""
        m = self.m
        seq = m.bos_emb.view(1, 1, -1) if prev_latent is None else prev_latent
        x = m.input_linear(seq) + self.cond
        h = m.out_norm(self.backbone(x))[:, -1]
        latent = noise + m.flow_net(h, noise, self.t_cond)
        return latent.view(1, 1, -1), m.out_eos(h)


def _conv_stream(conv, x, prev):
    xx = x if prev is None else torch.cat([prev, x], 2)
    y = F.conv1d(xx, conv.weight, conv.bias)
    if prev is None:
        return y, None
    return y, xx[:, :, xx.shape[2] - prev.shape[2] :]


def _convtr_stream(convtr, x, partial):
    y = F.conv_transpose1d(
        x, convtr.weight, convtr.bias, stride=convtr.stride, groups=convtr.groups
    )
    pt = partial.shape[2]
    y = torch.cat([y[:, :, :pt] + partial, y[:, :, pt:]], 2)
    new_partial = y[:, :, y.shape[2] - pt :]
    if convtr.bias is not None:
        new_partial = new_partial - convtr.bias.view(1, -1, 1)
    return y[:, :, : y.shape[2] - pt], new_partial


class MimiReference:
    """Streaming Mimi decoder; decode() takes any number of frames."""

    def __init__(self, model: Phonon):
        assert model.rope_layout == "interleaved"
        cfg, m = model.cfg, model.mimi
        self.cfg, self.m, self.lm = cfg, m, model.flow_lm
        z = lambda *s: torch.zeros(1, *s)  # noqa: E731
        self.up = z(cfg.mimi_dim, m.upsample.convtr.state_len)
        self.kv = [[None, None] for _ in range(cfg.mimi_layers)]
        self.offset = 0
        dec = m.decoder.model
        self.prev = {0: z(cfg.mimi_dim, dec[0].state_len)}
        self.partial = {}
        for j in range(len(cfg.ratios)):
            up, res = dec[2 + 3 * j], dec[3 + 3 * j]
            self.partial[j] = z(up.convtr.out_channels, up.state_len)
            self.prev[("res", j)] = z(up.convtr.out_channels, res.block[1].state_len)
        self.prev["final"] = z(cfg.n_filters, dec[-1].state_len)

    @torch.no_grad()
    def transformer(self, x):
        cfg = self.cfg
        t, d, h, hd = x.shape[1], cfg.mimi_dim, cfg.mimi_heads, cfg.mimi_head_dim
        pos = torch.arange(self.offset, self.offset + t)
        for i, layer in enumerate(self.m.decoder_transformer.transformer.layers):
            qkv = layer.self_attn.in_proj(layer.norm1(x)).view(1, t, 3, h, hd)
            q = rope_interleaved(qkv[:, :, 0], pos, cfg.mimi_max_period).transpose(1, 2)
            k = rope_interleaved(qkv[:, :, 1], pos, cfg.mimi_max_period).transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            pk, pv = self.kv[i]
            kk = k if pk is None else torch.cat([pk, k], 2)
            vv = v if pv is None else torch.cat([pv, v], 2)
            qi = torch.arange(t)[:, None] + kk.shape[2] - t
            kj = torch.arange(kk.shape[2])[None, :]
            ok = (kj <= qi) & (kj >= qi - cfg.mimi_context)
            mask = torch.where(ok, 0.0, float("-inf"))
            att = torch.softmax(q @ kk.transpose(-1, -2) / math.sqrt(hd) + mask, -1)
            o = (att @ vv).transpose(1, 2).reshape(1, t, d)
            self.kv[i] = [kk[:, :, -cfg.mimi_context :], vv[:, :, -cfg.mimi_context :]]
            x = x + layer.layer_scale_1.scale * layer.self_attn.out_proj(o)
            h2 = layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))
            x = x + layer.layer_scale_2.scale * h2
        self.offset += t
        return x

    @torch.no_grad()
    def decode(self, latent):
        """latent [1, T, ldim] -> pcm [1, 1, T * frame_samples]."""
        m, dec = self.m, self.m.decoder.model
        x = (latent * self.lm.emb_std + self.lm.emb_mean).transpose(1, 2)
        x = F.conv1d(x, m.quantizer.output_proj.weight)
        x, self.up = _convtr_stream(m.upsample.convtr.convtr, x, self.up)
        x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        x, self.prev[0] = _conv_stream(dec[0].conv, x, self.prev[0])
        for j in range(len(self.cfg.ratios)):
            up, res = dec[2 + 3 * j], dec[3 + 3 * j]
            x, self.partial[j] = _convtr_stream(up.convtr, F.elu(x), self.partial[j])
            h, self.prev[("res", j)] = _conv_stream(
                res.block[1].conv, F.elu(x), self.prev[("res", j)]
            )
            h, _ = _conv_stream(res.block[3].conv, F.elu(h), None)
            x = x + h
        x, self.prev["final"] = _conv_stream(dec[-1].conv, F.elu(x), self.prev["final"])
        return x
