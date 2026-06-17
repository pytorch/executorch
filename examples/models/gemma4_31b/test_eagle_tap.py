# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the gemma4-31B EAGLE-3 hidden-state tap.

Covers the tap-index convention (HF/vLLM: index 0 = embedding, index k = output
after decoder layer k-1), exact concatenation order/content, config validation
(including the runtime-mutation path), and that the default decode path is
unaffected by enabling the tap.
"""

import pytest
import torch

from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig


def tiny_config(num_layers=6, tap_layers=None) -> Gemma4_31BConfig:
    return Gemma4_31BConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_global_key_value_heads=1,
        global_head_dim=8,
        sliding_window=8,
        max_seq_len=32,
        eagle_tap_layers=tap_layers or [],
    )


def build(num_layers=6, tap_layers=None):
    torch.manual_seed(0)
    return Gemma4_31B(tiny_config(num_layers, tap_layers)).to(torch.float32).eval()


def reset_kv(model):
    """Zero the (stateful) KV caches so independent forwards don't couple."""
    for name, buf in model.named_buffers():
        if ".kv_cache." in name:
            buf.zero_()


def reference_states(model, tokens, input_pos):
    """Recompute _decode's per-index states: 0=embedding, k=after layer k-1."""
    x = model.embed_tokens(tokens) * model.embed_normalizer
    states = {0: x}
    sliding_mask, full_mask = model._build_masks(input_pos)
    for i, layer in enumerate(model.layers):
        x = layer(x, input_pos, sliding_mask, full_mask)
        states[i + 1] = x
    return states


def test_tap_off_does_not_change_logits():
    model = build(tap_layers=[1, 2, 3])
    T = 7
    tokens = torch.randint(0, 128, (1, T))
    pos = torch.arange(T)
    with torch.no_grad():
        reset_kv(model)
        logits_on, taps_on = model.forward_logits_taps(
            tokens, pos, last_logits_only=False
        )
        model.config.eagle_tap_layers = []
        reset_kv(model)
        logits_off, taps_off = model.forward_logits_taps(
            tokens, pos, last_logits_only=False
        )
    assert taps_off is None
    assert taps_on.shape == (1, T, 3 * model.config.hidden_size)
    torch.testing.assert_close(logits_on, logits_off)


@pytest.mark.parametrize(
    "num_layers,tap_layers",
    [
        (6, [0, 1, 3]),
        (60, [2, 30, 57]),
    ],
)
def test_tap_collects_exact_states_in_order(num_layers, tap_layers):
    model = build(num_layers=num_layers, tap_layers=tap_layers)
    T = 5
    tokens = torch.randint(0, 128, (1, T))
    pos = torch.arange(T)
    with torch.no_grad():
        reset_kv(model)
        _, taps = model.forward_logits_taps(tokens, pos)
        reset_kv(model)
        states = reference_states(model, tokens, pos)
    expected = torch.cat([states[i] for i in tap_layers], dim=-1)
    assert taps.shape == (1, T, len(tap_layers) * model.config.hidden_size)
    torch.testing.assert_close(taps, expected, rtol=0, atol=0)


def test_last_logits_only_default_matches_full():
    model = build(tap_layers=[1])
    T = 4
    tokens = torch.randint(0, 128, (1, T))
    pos = torch.arange(T)
    with torch.no_grad():
        reset_kv(model)
        full, _ = model.forward_logits_taps(tokens, pos, last_logits_only=False)
        reset_kv(model)
        last, _ = model.forward_logits_taps(tokens, pos)
    assert last.shape == (1, 1, model.config.vocab_size)
    torch.testing.assert_close(last[:, 0], full[:, -1])


@pytest.mark.parametrize("bad", [[99], [1, 1], [1.0, 2], [True], [3, 1]])
def test_invalid_tap_config_rejected(bad):
    with pytest.raises(ValueError):
        tiny_config(num_layers=6, tap_layers=bad)


def test_set_eagle_tap_layers_validates():
    model = build()
    model.set_eagle_tap_layers([0, 2, 4])
    assert model.config.eagle_tap_layers == [0, 2, 4]
    with pytest.raises(ValueError):
        model.set_eagle_tap_layers([4, 2])


def test_runtime_mutation_is_revalidated_in_decode():
    model = build(tap_layers=[1, 2])
    model.config.eagle_tap_layers = [True]
    tokens = torch.randint(0, 128, (1, 4))
    pos = torch.arange(4)
    with pytest.raises(ValueError):
        model.forward_logits_taps(tokens, pos, last_logits_only=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
