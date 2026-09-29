# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
from torch.nn import functional as F


def _layers():
    return importlib.import_module("examples.models.supertonic.model.layers")


def test_same_pad_1d_matches_edge_padding_in_onnx_graphs() -> None:
    values = torch.tensor([[[1.0, 2.0, 3.0]]])

    padded = _layers().SamePad1d(kernel_size=5)(values)

    torch.testing.assert_close(
        padded, torch.tensor([[[1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]]])
    )


def test_layer_norm_1d_normalizes_channels_last_and_restores_layout() -> None:
    values = torch.tensor([[[1.0, 3.0], [2.0, 5.0], [7.0, 11.0]]], dtype=torch.float32)
    layer = _layers().LayerNorm1d(3, eps=1e-6)
    with torch.no_grad():
        layer.norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5]))
        layer.norm.bias.copy_(torch.tensor([-1.0, 0.0, 1.0]))

    output = layer(values)

    expected = F.layer_norm(
        values.transpose(1, 2),
        (3,),
        layer.norm.weight,
        layer.norm.bias,
        eps=1e-6,
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)


def test_projection_wrappers_preserve_published_parameter_names() -> None:
    linear = _layers().LinearProjection(3, 4)
    convolution = _layers().Conv1dProjection(3, 4, kernel_size=1)

    assert set(linear.state_dict()) == {"linear.weight", "linear.bias"}
    assert set(convolution.state_dict()) == {"net.weight", "net.bias"}
    assert linear(torch.ones(2, 5, 3)).shape == (2, 5, 4)
    assert convolution(torch.ones(2, 3, 5)).shape == (2, 4, 5)


def test_convnext_block_matches_published_residual_sequence() -> None:
    block = _layers().ConvNeXtBlock(
        channels=2, kernel_size=3, expansion=2, layer_scale_init_value=1.0
    )
    with torch.no_grad():
        block.dwconv.weight.zero_()
        block.dwconv.weight[:, 0, 1] = 1.0
        block.dwconv.bias.zero_()
        block.norm.norm.weight.fill_(1.0)
        block.norm.norm.bias.zero_()
        block.pwconv1.weight.zero_()
        block.pwconv1.bias.zero_()
        block.pwconv1.weight[0, 0, 0] = 1.0
        block.pwconv1.weight[1, 1, 0] = 1.0
        block.pwconv2.weight.zero_()
        block.pwconv2.bias.zero_()
        block.pwconv2.weight[0, 0, 0] = 1.0
        block.pwconv2.weight[1, 1, 0] = 1.0
        block.gamma.fill_(1.0)
    values = torch.tensor([[[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]]])

    output = block(values)

    normalized = F.layer_norm(values.transpose(1, 2), (2,), eps=1e-6).transpose(1, 2)
    torch.testing.assert_close(output, values + F.gelu(normalized))


def test_convnext_block_applies_published_dilation() -> None:
    block = _layers().ConvNeXtBlock(
        channels=1,
        kernel_size=3,
        dilation=2,
        expansion=1,
        layer_scale_init_value=0.0,
    )

    assert block.pad.padding == (2, 2)
    assert block.dwconv.dilation == (2,)
    assert block(torch.ones(1, 1, 5)).shape == (1, 1, 5)


def test_convnext_masks_padding_before_and_after_each_block() -> None:
    torch.manual_seed(0)
    block = _layers().ConvNeXtBlock(channels=2, kernel_size=5)
    mask = torch.tensor([[[1.0, 1.0, 0.0]]])
    first = torch.randn(1, 2, 3)
    second = first.clone()
    second[:, :, -1] = 1000.0

    first_output = block(first, mask)
    second_output = block(second, mask)

    torch.testing.assert_close(first_output, second_output)
    torch.testing.assert_close(first_output[:, :, -1], torch.zeros(1, 2))


def test_convnext_stack_preserves_published_parameter_hierarchy() -> None:
    stack = _layers().ConvNeXt(channels=4, num_layers=2, kernel_size=5)

    assert "convnext.0.gamma" in stack.state_dict()
    assert "convnext.0.dwconv.weight" in stack.state_dict()
    assert "convnext.0.norm.norm.weight" in stack.state_dict()
    assert "convnext.1.pwconv2.bias" in stack.state_dict()
    assert stack(torch.ones(1, 4, 6)).shape == (1, 4, 6)


def test_multi_head_attention_matches_scaled_dot_product_attention() -> None:
    attention = _layers().MultiHeadAttention(channels=4, num_heads=2, bias=False)
    identity = torch.eye(4)
    with torch.no_grad():
        attention.W_query.linear.weight.copy_(identity)
        attention.W_key.linear.weight.copy_(identity)
        attention.W_value.linear.weight.copy_(identity)
        attention.out_fc.linear.weight.copy_(identity)
    values = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]])

    output = attention(values)

    heads = values.reshape(1, 2, 2, 2).transpose(1, 2)
    expected = F.scaled_dot_product_attention(heads, heads, heads)
    expected = expected.transpose(1, 2).reshape(1, 2, 4)
    torch.testing.assert_close(output, expected)


def test_multi_head_attention_applies_key_and_query_masks() -> None:
    torch.manual_seed(0)
    attention = _layers().MultiHeadAttention(channels=4, num_heads=2)
    query = torch.randn(1, 3, 4)
    context = torch.randn(1, 3, 4)
    changed_context = context.clone()
    changed_context[:, -1] = 1000.0
    key_mask = torch.tensor([[1.0, 1.0, 0.0]])
    query_mask = torch.tensor([[1.0, 1.0, 0.0]])

    output = attention(query, context, query_mask=query_mask, key_mask=key_mask)
    changed_output = attention(
        query, changed_context, query_mask=query_mask, key_mask=key_mask
    )

    torch.testing.assert_close(output, changed_output)
    torch.testing.assert_close(output[:, -1], torch.zeros(1, 4))


def test_multi_head_attention_returns_zero_when_all_keys_are_masked() -> None:
    torch.manual_seed(0)
    attention = _layers().MultiHeadAttention(channels=4, num_heads=2)
    query = torch.randn(2, 3, 4)
    context = torch.randn(2, 5, 4)
    key_mask = torch.zeros(2, 5)

    output = attention(query, context, key_mask=key_mask)

    torch.testing.assert_close(output, torch.zeros_like(output))
    assert torch.isfinite(output).all()


def test_multi_head_attention_preserves_published_projection_names() -> None:
    attention = _layers().MultiHeadAttention(channels=4, num_heads=2)

    assert {
        "W_query.linear.weight",
        "W_query.linear.bias",
        "W_key.linear.weight",
        "W_key.linear.bias",
        "W_value.linear.weight",
        "W_value.linear.bias",
        "out_fc.linear.weight",
        "out_fc.linear.bias",
    } == set(attention.state_dict())


def test_multi_head_attention_supports_cross_attention_projection_widths() -> None:
    attention = _layers().MultiHeadAttention(
        channels=4,
        context_channels=3,
        attention_channels=2,
        num_heads=1,
    )

    output = attention(torch.ones(1, 5, 4), torch.ones(1, 7, 3))

    assert output.shape == (1, 5, 4)
    assert attention.W_query.linear.weight.shape == (2, 4)
    assert attention.W_key.linear.weight.shape == (2, 3)
    assert attention.W_value.linear.weight.shape == (2, 3)
    assert attention.out_fc.linear.weight.shape == (4, 2)


def test_multi_head_attention_rejects_indivisible_attention_channels() -> None:
    with pytest.raises(
        ValueError,
        match="attention_channels must be divisible by num_heads",
    ):
        _layers().MultiHeadAttention(
            channels=4,
            attention_channels=5,
            num_heads=2,
        )


def test_add_conditioning_projects_and_broadcasts_over_sequence() -> None:
    conditioning = _layers().AddConditioning(condition_features=2, channels=3)
    with torch.no_grad():
        conditioning.linear.linear.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        conditioning.linear.linear.bias.zero_()
    values = torch.ones(1, 3, 4)
    condition = torch.tensor([[2.0, 3.0]])

    output = conditioning(values, condition)

    expected_condition = torch.tensor([[[2.0], [3.0], [5.0]]])
    torch.testing.assert_close(output, values + expected_condition)
    assert set(conditioning.state_dict()) == {
        "linear.linear.weight",
        "linear.linear.bias",
    }
