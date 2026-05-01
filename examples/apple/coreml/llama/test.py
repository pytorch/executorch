# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

sys.path.insert(0, ".")
import copy

import pytest
import torch
from export_static_llm_coreml import (
    _create_example_inputs,
    _resolve_cache_len,
    _resolve_per_layer_cache_lens,
)
from utils import replace_linear_with_split_linear

from executorch.examples.models.llama.model_args import ModelArgs


def get_split_model(
    model,
    out_target_split_size=1,
    out_max_splits=1,
    in_target_split_size=1,
    in_max_splits=1,
):
    model_copy = copy.deepcopy(model)
    replace_linear_with_split_linear(
        model_copy,
        out_target_split_size,
        out_max_splits,
        in_target_split_size,
        in_max_splits,
    )
    return model_copy


def test_split_model():
    inputs = torch.randn(10, 5, 1, 512)

    model = torch.nn.Sequential(*[torch.nn.Linear(512, 1024, bias=False)])
    model1 = get_split_model(model, 64, 2, 64, 1000)
    model2 = get_split_model(model, 64, 2, 64, 1)
    model3 = get_split_model(model, 64, 1, 64, 1000)

    assert torch.allclose(model(inputs), model1(inputs), atol=1e-5)
    assert torch.allclose(model(inputs), model2(inputs), atol=1e-5)
    assert torch.allclose(model(inputs), model3(inputs), atol=1e-5)


def test_resolve_cache_len_no_sliding_window():
    # Without --sliding_window the cache fills the rest of the context.
    assert _resolve_cache_len(1024, 32) == 992
    assert _resolve_cache_len(1024, 1) == 1023


def test_resolve_cache_len_with_sliding_window():
    # When the window is smaller than the remaining context the cache shrinks.
    assert _resolve_cache_len(8192, 32, sliding_window=4096) == 4096
    assert _resolve_cache_len(8192, 1, sliding_window=4096) == 4096


def test_resolve_cache_len_sliding_window_larger_than_context_is_a_no_op():
    # A user-provided window larger than the remaining context degenerates to
    # the no-window case, so users can safely set --sliding_window to a value
    # the model was trained with even when the export uses a shorter context.
    assert _resolve_cache_len(1024, 32, sliding_window=4096) == 992


def test_resolve_cache_len_rejects_non_positive_window():
    with pytest.raises(ValueError):
        _resolve_cache_len(1024, 32, sliding_window=0)
    with pytest.raises(ValueError):
        _resolve_cache_len(1024, 32, sliding_window=-1)


def test_create_example_inputs_with_sliding_window_shrinks_kv_cache():
    # Build a tiny ModelArgs that does not need a checkpoint or torchao.
    model_args = ModelArgs(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        vocab_size=128,
        max_context_len=1024,
        max_seq_len=1024,
    )
    max_context_len = 1024
    input_len = 32
    sliding_window = 64

    cache_len = _resolve_cache_len(max_context_len, input_len, sliding_window)
    assert cache_len == sliding_window

    example_inputs, returned_cache_len = _create_example_inputs(
        model_args,
        input_len,
        max_context_len,
        float_dtype=torch.float32,
        cache_len=cache_len,
    )
    assert returned_cache_len == sliding_window

    # The KV cache tensors live inside the kwargs dict at index 1 under
    # in_cache_state.  Walking that structure should find caches whose
    # sequence dimension equals the sliding window, not max_context_len.
    kwargs = example_inputs[1]
    in_cache_state = kwargs["in_cache_state"]
    cache_seq_dims = set()
    for per_kind in in_cache_state:  # (k_caches, v_caches)
        for cache_tensor in per_kind.values():
            cache_seq_dims.add(cache_tensor.size(-2))
    assert cache_seq_dims == {sliding_window}, (
        f"expected every KV cache to be sized to the sliding window {sliding_window}, "
        f"got {cache_seq_dims}"
    )

    # The attention mask covers (input_len + cache_len) along the last dim.
    masks = kwargs["masks"]
    assert sliding_window in masks
    assert masks[sliding_window].shape[-1] == input_len + sliding_window


def test_per_layer_cache_lens_uniform_when_no_pattern():
    # Without a pattern every layer gets the same cache length.
    out = _resolve_per_layer_cache_lens(
        n_layers=4, max_context_len=1024, input_len=32, sliding_window=64
    )
    assert out == [64, 64, 64, 64]


def test_per_layer_cache_lens_uniform_full_when_no_window():
    # No window at all is just `max_context_len - input_len` everywhere.
    out = _resolve_per_layer_cache_lens(
        n_layers=4, max_context_len=1024, input_len=32
    )
    assert out == [992, 992, 992, 992]


def test_per_layer_cache_lens_gemma4_e2b_pattern():
    # Gemma 4 E2B: 35 layers, P=5 → 4 sliding + 1 full repeated 7 times.
    out = _resolve_per_layer_cache_lens(
        n_layers=35,
        max_context_len=8192,
        input_len=32,
        sliding_window=4096,
        sliding_window_pattern=5,
    )
    full = 8192 - 32
    sliding = 4096
    assert len(out) == 35
    # Layers at 1-indexed positions 5, 10, 15, …, 35 are full.
    assert [out[i] for i in range(35)] == [
        full if (i + 1) % 5 == 0 else sliding for i in range(35)
    ]
    assert sum(1 for cl in out if cl == full) == 7
    assert sum(1 for cl in out if cl == sliding) == 28


def test_per_layer_cache_lens_gemma3_pattern():
    # Gemma 3 uses P=6 (5 sliding + 1 full).
    out = _resolve_per_layer_cache_lens(
        n_layers=12,
        max_context_len=2048,
        input_len=32,
        sliding_window=512,
        sliding_window_pattern=6,
    )
    full = 2048 - 32
    sliding = 512
    assert out == [
        sliding,
        sliding,
        sliding,
        sliding,
        sliding,
        full,
        sliding,
        sliding,
        sliding,
        sliding,
        sliding,
        full,
    ]


def test_per_layer_cache_lens_pattern_requires_sliding_window():
    with pytest.raises(ValueError):
        _resolve_per_layer_cache_lens(
            n_layers=8,
            max_context_len=1024,
            input_len=32,
            sliding_window=None,
            sliding_window_pattern=5,
        )


def test_per_layer_cache_lens_rejects_pattern_le_one():
    # P=1 would make every layer full and is almost certainly a typo, so
    # surface it rather than silently doing the no-pattern thing.
    with pytest.raises(ValueError):
        _resolve_per_layer_cache_lens(
            n_layers=8,
            max_context_len=1024,
            input_len=32,
            sliding_window=64,
            sliding_window_pattern=1,
        )


def test_per_layer_pattern_forward_pass_runs_end_to_end():
    """A tiny static-attention transformer must accept the heterogeneous
    cache shapes produced by `_resolve_per_layer_cache_lens` and run a
    forward pass without complaining about the mismatched cache sizes
    between sliding and full layers."""
    from executorch.examples.models.llama.llama_transformer import (
        construct_transformer,
    )
    from executorch.examples.models.llama.static_attention import (
        transform_attention_mha_to_static_attention,
    )

    args = ModelArgs(
        dim=64,
        n_layers=10,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        vocab_size=128,
        max_context_len=512,
        max_seq_len=512,
        attention_type="static_mha",
        attention_kwargs={"decompose_sdpa_in_mha": True},
    )
    model = construct_transformer(args).eval()
    model = transform_attention_mha_to_static_attention(
        model, split_mha=True, inplace=False
    ).eval()

    cache_lens = _resolve_per_layer_cache_lens(
        n_layers=10,
        max_context_len=512,
        input_len=32,
        sliding_window=64,
        sliding_window_pattern=5,
    )
    # 4 sliding + 1 full, twice.
    assert cache_lens == [64, 64, 64, 64, 480, 64, 64, 64, 64, 480]

    example_inputs, _ = _create_example_inputs(
        args, 32, 512, float_dtype=torch.float32, cache_len=cache_lens
    )
    with torch.no_grad():
        model(*example_inputs)


def test_create_example_inputs_with_per_layer_pattern_yields_two_cache_sizes():
    model_args = ModelArgs(
        dim=32,
        n_layers=10,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        vocab_size=128,
        max_context_len=1024,
        max_seq_len=1024,
    )
    max_context_len = 1024
    input_len = 32
    sliding_window = 64
    pattern = 5

    cache_lens = _resolve_per_layer_cache_lens(
        n_layers=model_args.n_layers,
        max_context_len=max_context_len,
        input_len=input_len,
        sliding_window=sliding_window,
        sliding_window_pattern=pattern,
    )

    example_inputs, _ = _create_example_inputs(
        model_args,
        input_len,
        max_context_len,
        float_dtype=torch.float32,
        cache_len=cache_lens,
    )

    in_cache_state = example_inputs[1]["in_cache_state"]
    seen = set()
    for per_kind in in_cache_state:
        for tensor in per_kind.values():
            seen.add(tensor.size(-2))
    full = max_context_len - input_len
    assert seen == {sliding_window, full}, (
        f"expected both {sliding_window} (sliding) and {full} (full) cache sizes, got {seen}"
    )

    # Both cache_len values get their own mask; (input_len + cache_len) per mask.
    masks = example_inputs[1]["masks"]
    assert set(masks.keys()) == {sliding_window, full}
    assert masks[sliding_window].shape[-1] == input_len + sliding_window
    assert masks[full].shape[-1] == input_len + full


if __name__ == "__main__":
    test_split_model()
    test_resolve_cache_len_no_sliding_window()
    test_resolve_cache_len_with_sliding_window()
    test_resolve_cache_len_sliding_window_larger_than_context_is_a_no_op()
    test_resolve_cache_len_rejects_non_positive_window()
    test_create_example_inputs_with_sliding_window_shrinks_kv_cache()
    test_per_layer_cache_lens_uniform_when_no_pattern()
    test_per_layer_cache_lens_uniform_full_when_no_window()
    test_per_layer_cache_lens_gemma4_e2b_pattern()
    test_per_layer_cache_lens_gemma3_pattern()
    test_per_layer_cache_lens_pattern_requires_sliding_window()
    test_per_layer_cache_lens_rejects_pattern_le_one()
    test_create_example_inputs_with_per_layer_pattern_yields_two_cache_sizes()
    test_per_layer_pattern_forward_pass_runs_end_to_end()
