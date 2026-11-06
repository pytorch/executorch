# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from executorch.backends.qualcomm._passes import I64toI32, TagQuantIO
from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
)
from executorch.backends.qualcomm.genai_pipeline.compilation.pass_policy import (
    build_component_pass_config,
    build_pass_config,
    get_pass_policy,
    resolve_num_sharding,
    TAGGER_DECODER,
    TAGGER_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    GRAPH_FORWARD,
    GRAPH_PREFILL_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.extension.llm.custom_ops.model_sharding import SplitGraph

_GEMMA4 = "gemma4-e2b"

# The decode graph's metadata, enough for both the shard boundaries and the
# quantized-IO shapes.
_DECODER_META = {
    "get_ar_len": 1,
    "get_head_dim": 64,
    "get_max_batch_size": 1,
    "get_max_context_len": 128,
    "get_n_layers": 4,
    "get_vocab_size": 32000,
}

_GEMMA4_META = {
    **_DECODER_META,
    "get_global_head_dim": 512,
    "get_n_self_layers": 3,
}

_QUANT_IO_DTYPES = {"kv_type": torch.uint8, "io_type": torch.uint16}
_ENCODER_IO_DTYPES = {"io_type": torch.uint16}


def _make_meta(**overrides):
    """Create decoder metadata with the required keys present."""
    meta = dict(_DECODER_META)
    meta.update(overrides)
    return meta


def _make_bundle(**overrides) -> GraphBundle:
    """Create a decoder GraphBundle with realistic metadata."""
    kwargs = {
        "module": MagicMock(name="module"),
        "inputs": (MagicMock(name="input"),),
        "meta": _make_meta(),
    }
    kwargs.update(overrides)
    return GraphBundle(**kwargs)


def _make_node(shape, op="call_function", name="node", target=None, users=()):
    """Create a node double whose shape and users the tagger can inspect."""
    node = MagicMock(name=name)
    node.op = op
    node.name = name
    node.target = target if target is not None else MagicMock(name=f"{name}_target")
    node.meta = {"val": torch.zeros(shape)}
    node.users = {user: None for user in users}
    return node


def _tagger(passes_job):
    """Return the quantized-IO tagger from a pass configuration."""
    return passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
        "get_quant_io_dtype_fn"
    ]


def _decoder_tagger(meta, control_args, quant_io_dtypes=_QUANT_IO_DTYPES):
    """Build a decoder pass configuration and return its tagger."""
    _, passes_job = build_pass_config(
        artifact_key=ARTIFACT_TEXT_DECODER,
        meta=meta,
        control_args=control_args,
        quant_io_dtypes=quant_io_dtypes,
    )
    return _tagger(passes_job)


class TestPassPolicyTable(unittest.TestCase):

    def test_taggers_differ_by_component(self):
        """The decoder matches shapes; an encoder only guards a split boundary."""
        self.assertEqual(get_pass_policy(ARTIFACT_TEXT_DECODER).tagger, TAGGER_DECODER)
        self.assertEqual(get_pass_policy(ARTIFACT_AUDIO_ENCODER).tagger, TAGGER_ENCODER)


class TestResolveNumSharding(unittest.TestCase):

    def test_keeps_a_count_within_the_layer_count(self):
        """A valid request is used as is."""
        graphs = {GRAPH_FORWARD: _make_bundle(meta=_make_meta(get_n_layers=8))}

        self.assertEqual(resolve_num_sharding(ARTIFACT_TEXT_DECODER, graphs, 4), 4)

    def test_clamps_a_count_above_the_layer_count_with_a_warning(self):
        """More shards than layers cannot be split, so the count is clamped."""
        graphs = {GRAPH_FORWARD: _make_bundle(meta=_make_meta(get_n_layers=2))}

        with self.assertLogs(
            "executorch.backends.qualcomm.genai_pipeline.compilation.pass_policy",
            level="WARNING",
        ):
            shards = resolve_num_sharding(ARTIFACT_TEXT_DECODER, graphs, 4)

        self.assertEqual(shards, 2)

    def test_rejects_sharding_without_a_layer_count(self):
        """Without the layer count there is nothing to check the request against."""
        graphs = {GRAPH_FORWARD: _make_bundle(meta={})}

        with self.assertRaises(ValueError) as ctx:
            resolve_num_sharding(ARTIFACT_TEXT_DECODER, graphs, 4)

        self.assertIn("get_n_layers", str(ctx.exception))


class TestSharding(unittest.TestCase):

    def test_inserts_the_splitting_pass_with_its_dependency_edges(self):
        """SplitGraph runs after FoldQDQ, and TagQuantIO after SplitGraph."""
        from executorch.backends.qualcomm._passes import FoldQDQ

        dep_table, passes_job = build_pass_config(
            artifact_key=ARTIFACT_TEXT_DECODER,
            meta=_make_meta(),
            control_args=ControlArgs(),
            num_sharding=4,
        )

        self.assertTrue(passes_job[SplitGraph][QCOM_PASS_ACTIVATE_KEY])
        self.assertEqual(dep_table[SplitGraph], [FoldQDQ])
        self.assertEqual(dep_table[TagQuantIO], [SplitGraph])

    def test_leaves_the_passes_alone_when_not_sharded(self):
        """One shard means no split, so the pass must not be inserted."""
        dep_table, passes_job = build_pass_config(
            artifact_key=ARTIFACT_TEXT_DECODER,
            meta=_make_meta(),
            control_args=ControlArgs(),
            num_sharding=1,
        )

        self.assertNotIn(SplitGraph, passes_job)
        self.assertNotIn(SplitGraph, dep_table)

    def test_places_shard_boundaries_from_the_layer_count(self):
        """The boundaries divide the layers, so the count is load-bearing."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_TEXT_DECODER,
            meta=_make_meta(get_n_layers=8),
            control_args=ControlArgs(),
            num_sharding=4,
        )

        kwargs = passes_job[SplitGraph][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]
        self.assertEqual(kwargs["shard_layers"], [0, 2, 4, 6])

    def test_rejects_sharding_without_the_layer_count(self):
        """Silently not splitting a sharded model compiles but is wrong."""
        with self.assertRaises(ValueError) as ctx:
            build_pass_config(
                artifact_key=ARTIFACT_TEXT_DECODER,
                meta={},
                control_args=ControlArgs(),
                num_sharding=4,
            )

        self.assertIn("get_n_layers", str(ctx.exception))

    def test_offsets_gemma4_cross_decoder_layers(self):
        """Its two decoders both count from zero, so one has to be offset."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_TEXT_DECODER,
            meta=dict(_GEMMA4_META),
            control_args=ControlArgs(decoder_model=_GEMMA4),
            num_sharding=2,
        )

        kwargs = passes_job[SplitGraph][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]
        self.assertEqual(
            kwargs["layer_prefix_offsets"],
            {
                "model.self_decoder.layers": 0,
                "model.cross_decoder.layers": _GEMMA4_META["get_n_self_layers"],
            },
        )

    def test_matches_gemma4_layers_under_both_module_paths(self):
        """A pattern matching only "layers" would find none of gemma4's."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_TEXT_DECODER,
            meta=dict(_GEMMA4_META),
            control_args=ControlArgs(decoder_model=_GEMMA4),
            num_sharding=2,
        )

        pattern = passes_job[SplitGraph][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]["pattern"]
        self.assertRegex("model.self_decoder.layers.0", pattern)
        self.assertRegex("model.cross_decoder.layers.1", pattern)

    def test_matches_encoder_layers_under_the_encoder_prefixes(self):
        """An encoder splits on its own module paths, not the decoder's."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_AUDIO_ENCODER,
            meta={"get_n_layers": 8},
            control_args=ControlArgs(),
            num_sharding=2,
        )

        pattern = passes_job[SplitGraph][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]["pattern"]
        self.assertRegex("encoder.layers.3", pattern)
        self.assertNotRegex("layers.3", pattern)


class TestEmbeddingFallback(unittest.TestCase):

    def test_skips_the_tokens_input_when_the_embedding_is_quantized(self):
        """A quantized embedding looks up in int64, so tokens must stay int64."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_TEXT_DECODER,
            meta=_make_meta(),
            control_args=ControlArgs(embedding_quantize="4,32"),
        )

        kwargs = passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]
        self.assertEqual(kwargs["skip_node"], {"tokens"})

    def test_narrows_every_input_by_default(self):
        """Without a quantized embedding there is nothing to preserve."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_TEXT_DECODER,
            meta=_make_meta(),
            control_args=ControlArgs(),
        )

        kwargs = passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]
        self.assertFalse(kwargs.get("skip_node"))


class TestQuantIOActivation(unittest.TestCase):

    def test_activates_tagging_when_quantization_published_dtypes(self):
        """The dtypes are the signal that there is a quantized boundary."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_TEXT_DECODER,
            meta=_make_meta(),
            control_args=ControlArgs(),
            quant_io_dtypes=_QUANT_IO_DTYPES,
        )

        self.assertTrue(passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY])

    def test_leaves_io_float32_when_quantization_was_skipped(self):
        """No dtypes means the graph was never quantized, so nothing is tagged."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_AUDIO_ENCODER,
            meta={"get_n_layers": 8},
            control_args=ControlArgs(),
            num_sharding=2,
            quant_io_dtypes=None,
        )

        self.assertFalse(passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY])

    def test_leaves_an_unsharded_encoder_untagged(self):
        """An encoder's tagging exists only to guard a split boundary."""
        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_AUDIO_ENCODER,
            meta={},
            control_args=ControlArgs(),
            quant_io_dtypes=_ENCODER_IO_DTYPES,
        )

        self.assertFalse(passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY])

    def test_rejects_tagging_without_the_shapes_it_matches_on(self):
        """A silently untagged tensor is an accuracy loss, not a failure."""
        with self.assertRaises(ValueError) as ctx:
            build_pass_config(
                artifact_key=ARTIFACT_TEXT_DECODER,
                meta={"get_ar_len": 1},
                control_args=ControlArgs(),
                quant_io_dtypes=_QUANT_IO_DTYPES,
            )

        self.assertIn("get_vocab_size", str(ctx.exception))

    def test_rejects_decoder_dtypes_without_a_kv_type(self):
        """The decoder tags its KV caches, so an activation width alone is wrong."""
        with self.assertRaises(ValueError) as ctx:
            build_pass_config(
                artifact_key=ARTIFACT_TEXT_DECODER,
                meta=_make_meta(),
                control_args=ControlArgs(),
                quant_io_dtypes=_ENCODER_IO_DTYPES,
            )

        self.assertIn("kv_type", str(ctx.exception))


class TestTokenEmbeddingBoundary(unittest.TestCase):

    def test_tags_with_the_decoder_dtypes_when_its_own_are_absent(self):
        """Legacy binds the decoder's dtypes, so the embedding needs no recipe."""
        decoder_graphs = {
            "kv_forward": _make_bundle(quant_io_dtypes=_QUANT_IO_DTYPES),
        }
        graphs = {
            TOK_EMBEDDING_GRAPH_NAMES[0]: _make_bundle(meta={}, quant_io_dtypes=None),
        }

        _, passes_jobs = build_component_pass_config(
            artifact_key=ARTIFACT_TOK_EMBEDDING,
            graphs=graphs,
            control_args=ControlArgs(),
            decoder_graphs=decoder_graphs,
        )

        passes_job = passes_jobs[TOK_EMBEDDING_GRAPH_NAMES[0]]
        self.assertTrue(passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY])
        logits = _make_node((1, 1, 32000), op="placeholder")
        self.assertEqual(_tagger(passes_job)(logits), torch.uint16)

    def test_rejects_a_missing_decode_graph(self):
        """Without it, the embedding boundary would stay float32 without a signal."""
        graphs = {TOK_EMBEDDING_GRAPH_NAMES[0]: _make_bundle(meta={})}

        with self.assertRaises(ValueError) as ctx:
            build_component_pass_config(
                artifact_key=ARTIFACT_TOK_EMBEDDING,
                graphs=graphs,
                control_args=ControlArgs(),
                decoder_graphs={GRAPH_PREFILL_FORWARD: _make_bundle()},
            )

        self.assertIn("kv_forward", str(ctx.exception))


class TestDecoderTagger(unittest.TestCase):

    def test_tags_a_logits_output_with_the_io_dtype(self):
        """The logits leave the graph quantized, at the recipe's width."""
        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        node = _make_node((1, 1, 32000), op="placeholder")

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["io_type"])

    def test_tags_a_kv_cache_input_with_the_kv_dtype(self):
        """KV caches use their own width, which the runtime shares with prefill."""
        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        user = _make_node((1, 1, 64, 128))
        node = _make_node(
            (1, 1, 64, 128),
            op="placeholder",
            name="args_0",
            users=(user,),
        )

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["kv_type"])

    def test_tags_the_attention_mask_with_the_io_dtype(self):
        """The mask is a graph input and broadcasts across heads."""
        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        node = _make_node((1, 1, 1, 128), op="placeholder")

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["io_type"])

    def test_tags_a_tensor_crossing_a_shard_boundary(self):
        """The fallback op marks the split, so its operands stay quantized."""
        from executorch.exir.dialects._ops import ops as exir_ops

        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        consumer = _make_node((1, 1, 8), target=exir_ops.edge.llama.fallback.default)
        node = _make_node((1, 1, 8), users=(consumer,))

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["io_type"])

    def test_leaves_an_unrelated_tensor_alone(self):
        """Only the boundary tensors are tagged; the rest lower as usual."""
        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        node = _make_node((3, 7))

        self.assertIsNone(tag_ios(node))

    def test_tags_a_kv_cache_output_with_the_kv_dtype(self):
        """The updated caches leave quantized, at the same width they entered.

        Prefill and decode share one RPC buffer with no requantization, so an
        untagged cache output would silently change the scale the other mode
        reads.
        """
        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        node = _make_node((1, 1, 1, 64), users=(_make_node((1,), op="output"),))

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["kv_type"])

    def test_tags_a_logits_graph_output_with_the_io_dtype(self):
        """The logits are tagged on the way out as well as in."""
        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        node = _make_node((1, 1, 32000), users=(_make_node((1,), op="output"),))

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["io_type"])

    def test_tags_the_rope_frequency_slices(self):
        """Sharding splits freq_sin / freq_cos out as select ops."""
        from executorch.exir.dialects._ops import ops as exir_ops

        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        node = _make_node((1, 32), target=exir_ops.edge.aten.select.int)

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["io_type"])

    def test_tags_gemma4_kv_caches_of_the_global_head_dim(self):
        """Gemma4's full-attention layers use a second head dim."""
        tag_ios = _decoder_tagger(
            dict(_GEMMA4_META), ControlArgs(decoder_model=_GEMMA4)
        )
        node = _make_node((1, 1, 1, 512), users=(_make_node((1,), op="output"),))

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["kv_type"])

    def test_tags_gemma4_per_layer_inputs_by_stack_trace(self):
        """Matched on source text, so this is coupled to the model's forward."""
        tag_ios = _decoder_tagger(
            dict(_GEMMA4_META), ControlArgs(decoder_model=_GEMMA4)
        )
        node = _make_node((3, 7))
        node.meta["stack_trace"] = "x = per_layer_inputs[i]"

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["io_type"])

    def test_ignores_gemma4_rules_for_other_models(self):
        """The default layout has no extra rules, so the same node is untouched."""
        tag_ios = _decoder_tagger(_make_meta(), ControlArgs())
        node = _make_node((3, 7))
        node.meta["stack_trace"] = "x = per_layer_inputs[i]"

        self.assertIsNone(tag_ios(node))

    def test_tags_gemma4_donor_kv_by_source_name(self):
        """YOCO shared layers consume full_k / full_v, matched by source name."""
        tag_ios = _decoder_tagger(
            dict(_GEMMA4_META), ControlArgs(decoder_model=_GEMMA4)
        )
        node = _make_node((3, 7))
        node.meta["from_node"] = [SimpleNamespace(name="full_k_0", from_node=None)]

        self.assertEqual(tag_ios(node), _QUANT_IO_DTYPES["io_type"])

    def test_leaves_a_node_with_unrelated_sources_alone(self):
        """The source-name walk recurses but matches nothing here."""
        tag_ios = _decoder_tagger(
            dict(_GEMMA4_META), ControlArgs(decoder_model=_GEMMA4)
        )
        node = _make_node((3, 7))
        child = SimpleNamespace(name="unrelated", from_node=None)
        node.meta["from_node"] = [SimpleNamespace(name="parent", from_node=[child])]

        self.assertIsNone(tag_ios(node))


class TestEncoderTagger(unittest.TestCase):

    def test_tags_a_sharding_boundary_tensor_with_the_recipe_io_dtype(self):
        """The boundary takes the encoder recipe's activation dtype (#22924)."""
        from executorch.exir.dialects._ops import ops as exir_ops

        _, passes_job = build_pass_config(
            artifact_key=ARTIFACT_AUDIO_ENCODER,
            meta={"get_n_layers": 8},
            control_args=ControlArgs(),
            num_sharding=2,
            quant_io_dtypes=_ENCODER_IO_DTYPES,
        )
        consumer = _make_node((1, 8), target=exir_ops.edge.llama.fallback.default)
        node = _make_node((1, 8), users=(consumer,))

        self.assertEqual(_tagger(passes_job)(node), torch.uint16)


if __name__ == "__main__":
    unittest.main()
