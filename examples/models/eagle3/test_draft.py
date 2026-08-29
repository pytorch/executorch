# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the EAGLE-3 draft head and its checkpoint adapter."""

import json
import os

import pytest
import torch

from executorch.examples.models.eagle3.draft import Eagle3Config, Eagle3Draft


def tiny_config(norm_before_residual=True, has_own_embed=True) -> Eagle3Config:
    return Eagle3Config(
        hidden_size=32,
        target_hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        draft_vocab_size=16,
        target_vocab_size=40,
        aux_hidden_state_layers=[0, 1, 2],
        norm_before_residual=norm_before_residual,
        has_own_embed=has_own_embed,
    )


@pytest.mark.parametrize("norm_before_residual", [True, False])
@pytest.mark.parametrize("has_own_embed", [True, False])
def test_tiny_forward_shapes(norm_before_residual, has_own_embed):
    torch.manual_seed(0)
    cfg = tiny_config(norm_before_residual, has_own_embed)
    model = Eagle3Draft(cfg).to(torch.float32).eval()
    B, T = 1, 5

    aux = torch.randn(B, T, len(cfg.aux_hidden_state_layers) * cfg.target_hidden_size)
    feat = model.fuse(aux)
    assert feat.shape == (B, T, cfg.hidden_size)

    if has_own_embed:
        emb = model.embed(torch.randint(0, cfg.target_vocab_size, (T,))).unsqueeze(0)
        assert emb.shape == (B, T, cfg.hidden_size)
    else:
        emb = torch.randn(B, T, cfg.hidden_size)

    with torch.no_grad():
        logits, g = model(emb, feat, torch.arange(T))
    assert logits.shape == (B, T, cfg.draft_vocab_size)
    assert g.shape == (B, T, cfg.hidden_size)
    assert torch.isfinite(logits).all() and torch.isfinite(g).all()


def test_norm_before_residual_changes_output():
    # Check residual-path wiring, not only output shape.
    B, T = 1, 5
    aux = torch.randn(B, T, 3 * 32)
    emb = torch.randn(B, T, 32)
    outs = []
    for nbr in (True, False):
        torch.manual_seed(1)  # identical weights, only the flag differs
        model = (
            Eagle3Draft(tiny_config(norm_before_residual=nbr)).to(torch.float32).eval()
        )
        with torch.no_grad():
            _, g = model(emb, model.fuse(aux), torch.arange(T))
        outs.append(g)
    assert not torch.allclose(outs[0], outs[1]), "norm_before_residual had no effect"


def test_draft_to_target_mapping():
    model = Eagle3Draft(tiny_config()).eval()
    model.d2t.copy_(torch.arange(model.config.draft_vocab_size))  # offset = id
    ids = torch.tensor([0, 3, 7])
    assert torch.equal(model.draft_to_target(ids), ids + ids)


def test_embed_requires_own_embed():
    model = Eagle3Draft(tiny_config(has_own_embed=False)).eval()
    assert not hasattr(model, "embed_tokens")
    with pytest.raises(RuntimeError, match="no own embed_tokens"):
        model.embed(torch.tensor([0, 1, 2]))


def test_inv_freq_stays_fp32_under_assign_load():
    cfg = tiny_config()
    model = Eagle3Draft(cfg)
    assert model.midlayer.self_attn.inv_freq.dtype == torch.float32
    sd = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    model.load_state_dict(sd, strict=True, assign=True)
    assert model.midlayer.self_attn.inv_freq.dtype == torch.float32
    assert model.fc.weight.dtype == torch.bfloat16


def _write_checkpoint(model_dir, cfg, *, sharded=False, norm_before_fc=False):
    """Write a tiny speculators-format checkpoint (config.json + safetensors)."""
    from safetensors.torch import save_file

    os.makedirs(model_dir, exist_ok=True)
    torch.manual_seed(2)
    src = Eagle3Draft(cfg)
    disk = {
        k.replace("midlayer.", "layers.0."): v.clone().contiguous()
        for k, v in src.state_dict().items()
    }
    d2t = torch.arange(cfg.draft_vocab_size, dtype=torch.int64)
    t2d = torch.zeros(cfg.target_vocab_size, dtype=torch.bool)
    t2d[: cfg.draft_vocab_size] = True
    disk["d2t"] = d2t
    disk["t2d"] = t2d

    config = {
        "draft_vocab_size": cfg.draft_vocab_size,
        "target_hidden_size": cfg.target_hidden_size,
        "eagle_aux_hidden_state_layer_ids": cfg.aux_hidden_state_layers,
        "norm_before_residual": cfg.norm_before_residual,
        "norm_before_fc": norm_before_fc,
        "transformer_layer_config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "vocab_size": cfg.target_vocab_size,
            "rms_norm_eps": cfg.rms_norm_eps,
            "rope_parameters": {"rope_theta": cfg.rope_theta},
        },
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    if not sharded:
        save_file(disk, os.path.join(model_dir, "model.safetensors"))
    else:
        keys = list(disk)
        half = len(keys) // 2
        s1 = {k: disk[k] for k in keys[:half]}
        s2 = {k: disk[k] for k in keys[half:]}
        save_file(s1, os.path.join(model_dir, "model-00001-of-00002.safetensors"))
        save_file(s2, os.path.join(model_dir, "model-00002-of-00002.safetensors"))
        weight_map = {k: "model-00001-of-00002.safetensors" for k in s1}
        weight_map.update({k: "model-00002-of-00002.safetensors" for k in s2})
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)
    return src, d2t, t2d


@pytest.mark.parametrize("sharded", [False, True])
def test_from_checkpoint_roundtrip(tmp_path, sharded):
    cfg = tiny_config()
    src, d2t, t2d = _write_checkpoint(str(tmp_path), cfg, sharded=sharded)

    model, loaded_cfg = Eagle3Draft.from_checkpoint(
        str(tmp_path), device="cpu", dtype=torch.float32
    )
    assert loaded_cfg.has_own_embed
    assert loaded_cfg.aux_hidden_state_layers == cfg.aux_hidden_state_layers
    assert loaded_cfg.target_vocab_size == cfg.target_vocab_size
    torch.testing.assert_close(
        model.midlayer.self_attn.q_proj.weight, src.midlayer.self_attn.q_proj.weight
    )
    torch.testing.assert_close(model.fc.weight, src.fc.weight)
    assert torch.equal(model.d2t, d2t)
    assert torch.equal(model.t2d, t2d)
    assert model.midlayer.self_attn.inv_freq.dtype == torch.float32
    T = 4
    feat = model.fuse(torch.randn(1, T, 3 * cfg.target_hidden_size))
    emb = model.embed(torch.randint(0, cfg.target_vocab_size, (T,))).unsqueeze(0)
    logits, g = model(emb, feat, torch.arange(T))
    assert logits.shape == (1, T, cfg.draft_vocab_size)


def test_from_checkpoint_rejects_norm_before_fc(tmp_path):
    cfg = tiny_config()
    _write_checkpoint(str(tmp_path), cfg, norm_before_fc=True)
    with pytest.raises(ValueError, match="norm_before_fc"):
        Eagle3Draft.from_checkpoint(str(tmp_path), device="cpu", dtype=torch.float32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
