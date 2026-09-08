# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest import TestCase

import torch

from executorch.backends.vulkan.patterns.sdpa import (
    CausalSDPAMatch,
    NonCausalSDPAMatch,
    replace_custom_sdpa_with_noncausal_sdpa,
)
from executorch.exir.dialects._ops import ops as exir_ops


class _FakeOp:
    __name__ = "fake_op"

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args, **kwargs):
        raise AssertionError("Fake operator must not execute")

    def name(self) -> str:
        return self._name


CUSTOM_SDPA = _FakeOp("llama::custom_sdpa")
UPDATE_CACHE = _FakeOp("llama::update_cache")


def _make_custom_sdpa_node(
    argument_style: str = "full",
    *,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
    scale=None,
    update_key=True,
    update_value=True,
):
    graph = torch.fx.Graph()
    query = graph.placeholder("query")
    key = graph.placeholder("key")
    value = graph.placeholder("value")
    start_pos = graph.placeholder("start_pos")

    if argument_style == "full":
        args = (
            query,
            key,
            value,
            start_pos,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
        )
        kwargs = {}
    elif argument_style == "shortened":
        args = (query, key, value, start_pos)
        kwargs = {"is_causal": is_causal}
    elif argument_style == "keyword":
        args = ()
        kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "start_pos": start_pos,
            "attn_mask": attn_mask,
            "drpout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
        }
    elif argument_style == "defaults":
        args = (query, key, value, start_pos)
        kwargs = {}
    elif argument_style == "malformed":
        args = (query, key)
        kwargs = {}
    else:
        raise AssertionError(f"Unknown argument style: {argument_style}")

    custom_sdpa = graph.call_function(CUSTOM_SDPA, args=args, kwargs=kwargs)
    key_projection = graph.placeholder("key_projection")
    value_projection = graph.placeholder("value_projection")
    if update_key:
        graph.call_function(UPDATE_CACHE, args=(key_projection, key, start_pos))
    if update_value:
        graph.call_function(UPDATE_CACHE, args=(value_projection, value, start_pos))
    return custom_sdpa


class TestCausalSDPAMatch(TestCase):
    def test_matches_full_positional_arguments(self) -> None:
        self.assertTrue(CausalSDPAMatch(_make_custom_sdpa_node()).match_found)

    def test_matches_shortened_arguments_with_causal_keyword(self) -> None:
        node = _make_custom_sdpa_node(argument_style="shortened")
        self.assertTrue(CausalSDPAMatch(node).match_found)

    def test_matches_keyword_arguments(self) -> None:
        node = _make_custom_sdpa_node(argument_style="keyword")
        self.assertTrue(CausalSDPAMatch(node).match_found)

    def test_rejects_omitted_non_causal_default(self) -> None:
        node = _make_custom_sdpa_node(argument_style="defaults")
        self.assertFalse(CausalSDPAMatch(node).match_found)

    def test_rejects_malformed_required_arguments(self) -> None:
        node = _make_custom_sdpa_node(argument_style="malformed")
        self.assertFalse(CausalSDPAMatch(node).match_found)

    def test_rejects_incomplete_cache_updates(self) -> None:
        key_only = _make_custom_sdpa_node(update_value=False)
        value_only = _make_custom_sdpa_node(update_key=False)
        self.assertFalse(CausalSDPAMatch(key_only).match_found)
        self.assertFalse(CausalSDPAMatch(value_only).match_found)

    def test_rejects_non_causal_semantics(self) -> None:
        cases = (
            {"attn_mask": object()},
            {"dropout_p": 0.1},
            {"is_causal": False},
            {"scale": 0.125},
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                node = _make_custom_sdpa_node(**kwargs)
                self.assertFalse(CausalSDPAMatch(node).match_found)


def _make_noncausal_sdpa_node(
    *,
    q_shape=(1, 4, 32, 128),
    k_shape=(1, 128, 8, 128),
    v_shape=(1, 128, 8, 128),
    mask_shape=(4, 128),
    dtype=torch.float32,
    mask_dtype=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    graph = torch.fx.Graph()

    def placeholder(name, shape, tensor_dtype=dtype):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=tensor_dtype)
        return node

    query = placeholder("query", q_shape)
    key = placeholder("key", k_shape)
    value = placeholder("value", v_shape)
    start_pos = graph.placeholder("start_pos")
    attn_mask = None
    if mask_shape is not None:
        attn_mask = placeholder("attn_mask", mask_shape, mask_dtype or dtype)

    custom_sdpa = graph.call_function(
        CUSTOM_SDPA,
        args=(
            query,
            key,
            value,
            start_pos,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
        ),
    )
    custom_sdpa.meta["val"] = torch.empty(q_shape, dtype=dtype)
    graph.output((custom_sdpa,))
    return SimpleNamespace(graph=graph), custom_sdpa


class TestNonCausalSDPAMatch(TestCase):
    def test_matches_equal_head_gqa_and_broadcast_masks(self) -> None:
        cases = (
            _make_noncausal_sdpa_node(
                k_shape=(1, 128, 32, 128),
                v_shape=(1, 128, 32, 128),
            ),
            _make_noncausal_sdpa_node(),
            _make_noncausal_sdpa_node(mask_shape=(1, 1, 4, 128)),
            _make_noncausal_sdpa_node(mask_shape=None, scale=0.125),
        )
        for _, node in cases:
            with self.subTest(node=node):
                self.assertTrue(NonCausalSDPAMatch(node).match_found)

    def test_rejects_unsupported_semantics_and_shapes(self) -> None:
        cases = (
            _make_noncausal_sdpa_node(is_causal=True),
            _make_noncausal_sdpa_node(dropout_p=0.1),
            _make_noncausal_sdpa_node(q_shape=(1, 4, 30, 128)),
            _make_noncausal_sdpa_node(mask_shape=(2, 4, 128)),
            _make_noncausal_sdpa_node(mask_dtype=torch.float16),
        )
        for _, node in cases:
            with self.subTest(node=node):
                self.assertFalse(NonCausalSDPAMatch(node).match_found)

    def test_replacement_uses_fused_sdpa_without_expanding_kv_heads(self) -> None:
        graph_module, node = _make_noncausal_sdpa_node()
        match = NonCausalSDPAMatch(node)
        self.assertTrue(match.match_found)

        replace_custom_sdpa_with_noncausal_sdpa(object(), graph_module, match)

        fused_nodes = [
            graph_node
            for graph_node in graph_module.graph.nodes
            if graph_node.target == exir_ops.edge.et_vk.sdpa.default
        ]
        self.assertEqual(len(fused_nodes), 1)
        fused_node = fused_nodes[0]
        self.assertEqual(fused_node.meta["val"].shape, (1, 32, 4, 128))
        self.assertTrue(fused_node.meta["val"].is_contiguous())
        self.assertEqual(fused_node.args[1].meta["val"].shape, (1, 8, 128, 128))
        self.assertEqual(fused_node.args[2].meta["val"].shape, (1, 8, 128, 128))
        self.assertTrue(fused_node.args[0].meta["val"].is_contiguous())
        self.assertTrue(fused_node.args[1].meta["val"].is_contiguous())
        self.assertTrue(fused_node.args[2].meta["val"].is_contiguous())
        self.assertFalse(
            any(
                "repeat" in str(graph_node.target)
                for graph_node in graph_module.graph.nodes
            )
        )
