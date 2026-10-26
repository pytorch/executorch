import unittest

import torch

from executorch.backends.vulkan.patterns.sdpa import (
    CausalSDPAMatch,
    is_update_cache_node,
    replace_custom_sdpa_with_causal_sdpa,
)
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401


class CausalSDPAPatternTest(unittest.TestCase):
    def test_rejects_shared_kv_without_cache_updates(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        start_pos = graph.placeholder("start_pos")
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, k_cache, v_cache, start_pos, None, 0.0, True, 1.0),
        )
        graph.output(custom_sdpa)

        match = CausalSDPAMatch(custom_sdpa)

        self.assertFalse(match.match_found)

    def test_preserves_explicit_scale(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        k = graph.placeholder("k")
        v = graph.placeholder("v")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        start_pos = graph.placeholder("start_pos")
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(k, k_cache, start_pos),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(v, v_cache, start_pos),
        )
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, k_cache, v_cache, start_pos, None, 0.0, True, 1.0),
        )
        graph.output(custom_sdpa)

        match = CausalSDPAMatch(custom_sdpa)

        self.assertTrue(match.match_found)
        self.assertEqual(match.scale_node, 1.0)

    def test_rejects_cache_shared_with_later_sdpa(self) -> None:
        graph = torch.fx.Graph()
        q_donor = graph.placeholder("q_donor")
        q_shared = graph.placeholder("q_shared")
        k = graph.placeholder("k")
        v = graph.placeholder("v")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        start_pos = graph.placeholder("start_pos")
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(k, k_cache, start_pos),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(v, v_cache, start_pos),
        )
        donor_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q_donor, k_cache, v_cache, start_pos, None, 0.0, True, 1.0),
        )
        shared_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q_shared, k_cache, v_cache, start_pos, None, 0.0, True, 1.0),
        )
        graph.output((donor_sdpa, shared_sdpa))

        match = CausalSDPAMatch(donor_sdpa)

        self.assertFalse(match.match_found)

    def test_rejects_live_cache_update_result(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        k = graph.placeholder("k")
        v = graph.placeholder("v")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        start_pos = graph.placeholder("start_pos")
        updated_k = graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(k, k_cache, start_pos),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(v, v_cache, start_pos),
        )
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, k_cache, v_cache, start_pos, None, 0.0, True, 1.0),
        )
        graph.output((custom_sdpa, updated_k))

        match = CausalSDPAMatch(custom_sdpa)

        self.assertFalse(match.match_found)

    def test_rejects_cache_used_as_update_source(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        v = graph.placeholder("v")
        k_cache = graph.placeholder("k_cache")
        other_k_cache = graph.placeholder("other_k_cache")
        v_cache = graph.placeholder("v_cache")
        start_pos = graph.placeholder("start_pos")
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(k_cache, other_k_cache, start_pos),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(v, v_cache, start_pos),
        )
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, k_cache, v_cache, start_pos, None, 0.0, True, 1.0),
        )
        graph.output(custom_sdpa)

        match = CausalSDPAMatch(custom_sdpa)

        self.assertFalse(match.match_found)

    def test_rejects_mismatched_cache_update_position(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        k = graph.placeholder("k")
        v = graph.placeholder("v")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        update_pos = graph.placeholder("update_pos")
        attention_pos = graph.placeholder("attention_pos")
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(k, k_cache, update_pos),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(v, v_cache, update_pos),
        )
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, k_cache, v_cache, attention_pos, None, 0.0, True, 1.0),
        )
        graph.output(custom_sdpa)

        match = CausalSDPAMatch(custom_sdpa)

        self.assertFalse(match.match_found)

    def test_rejects_cache_updates_after_attention(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        k = graph.placeholder("k")
        v = graph.placeholder("v")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        start_pos = graph.placeholder("start_pos")
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, k_cache, v_cache, start_pos, None, 0.0, True, 1.0),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(k, k_cache, start_pos),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(v, v_cache, start_pos),
        )
        graph.output(custom_sdpa)

        match = CausalSDPAMatch(custom_sdpa)

        self.assertFalse(match.match_found)

    def test_rejects_aliased_key_and_value_caches(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        projected = graph.placeholder("projected")
        cache = graph.placeholder("cache")
        start_pos = graph.placeholder("start_pos")
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(projected, cache, start_pos),
        )
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, cache, cache, start_pos, None, 0.0, True, 1.0),
        )
        graph.output(custom_sdpa)

        match = CausalSDPAMatch(custom_sdpa)

        self.assertFalse(match.match_found)

    def test_rejects_cache_updated_from_itself(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        v = graph.placeholder("v")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        start_pos = graph.placeholder("start_pos")
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(k_cache, k_cache, start_pos),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(v, v_cache, start_pos),
        )
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, k_cache, v_cache, start_pos, None, 0.0, True, 1.0),
        )
        graph.output(custom_sdpa)

        match = CausalSDPAMatch(custom_sdpa)

        self.assertFalse(match.match_found)

    def test_replacement_preserves_arguments_and_erases_updates(self) -> None:
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        k = graph.placeholder("k")
        v = graph.placeholder("v")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        start_pos = graph.placeholder("start_pos")
        scale = graph.placeholder("scale")
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(k, k_cache, start_pos),
        )
        graph.call_function(
            torch.ops.llama.update_cache.default,
            args=(v, v_cache, start_pos),
        )
        custom_sdpa = graph.call_function(
            torch.ops.llama.custom_sdpa.default,
            args=(q, k_cache, v_cache, start_pos, None, 0.0, True, scale),
        )
        custom_sdpa.meta["val"] = None
        graph.output(custom_sdpa)
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
        match = CausalSDPAMatch(custom_sdpa)

        self.assertTrue(match.match_found)
        replace_custom_sdpa_with_causal_sdpa(None, graph_module, match)
        graph_module.graph.lint()

        fused_nodes = [
            node
            for node in graph_module.graph.nodes
            if node.target is torch.ops.llama.sdpa_with_kv_cache.default
        ]
        self.assertEqual(len(fused_nodes), 1)
        self.assertIs(fused_nodes[0].args[5], start_pos)
        self.assertIs(fused_nodes[0].args[10], scale)
        self.assertFalse(
            any(is_update_cache_node(node) for node in graph_module.graph.nodes)
        )
