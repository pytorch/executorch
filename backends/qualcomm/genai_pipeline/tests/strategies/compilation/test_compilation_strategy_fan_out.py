# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    GRAPH_KV_FORWARD,
    GRAPH_PREFILL_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilationResult,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.executorch_compilation_strategy import (
    ExecuTorchCompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
    TEST_BACKEND_TYPE,
    TEST_MODEL_NAME,
    TEST_SOC_CHIPSET,
)


def _make_group_adapter():
    """Create a mock adapter whose group call names the artifact it was given."""
    adapter = MagicMock()
    adapter.compile_graph_group.side_effect = lambda **kwargs: CompilationResult(
        artifact_paths={
            kwargs["artifact_key"]: Path(f"/tmp/{kwargs['file_name']}.pte")
        },
    )
    return adapter


# Enough metadata for the decoder's shard boundaries and quantized-IO shapes.
_DECODER_META = {
    "get_ar_len": 1,
    "get_head_dim": 64,
    "get_max_batch_size": 1,
    "get_max_context_len": 128,
    "get_n_layers": 4,
    "get_vocab_size": 32000,
}


def _make_bundle(**overrides) -> GraphBundle:
    """Create a valid GraphBundle, carrying metadata unless overridden.

    ``meta`` is populated by default because a graph without it cannot be sharded
    or tagged: ``pass_policy`` raises rather than silently skipping, so a bundle
    with no metadata is not a realistic input.
    """
    kwargs = {
        "module": MagicMock(name="module"),
        "inputs": (MagicMock(name="input"),),
        "meta": dict(_DECODER_META),
    }
    kwargs.update(overrides)
    return GraphBundle(**kwargs)


def _make_decoder_graphs(**overrides):
    """Build the two-graph decoder component."""
    return {name: _make_bundle(**overrides) for name in DECODER_GRAPH_NAMES}


def _make_input_config(graphs, **overrides):
    """Create a component-keyed CompilationInputConfig."""
    defaults = {
        "soc_model": TEST_SOC_CHIPSET,
        "backend_type": TEST_BACKEND_TYPE,
        "artifact_dir": Path("/tmp/artifacts"),
        "graphs": graphs,
    }
    defaults.update(overrides)
    return CompilationInputConfig(**defaults)


class TestCompilationStrategyFanOut(unittest.TestCase):

    def test_makes_one_group_call_per_component(self):
        """Each component becomes its own .pte, so each gets its own call."""
        graphs = {
            ARTIFACT_TEXT_DECODER: _make_decoder_graphs(),
            ARTIFACT_TOK_EMBEDDING: {
                name: _make_bundle() for name in TOK_EMBEDDING_GRAPH_NAMES
            },
        }
        adapter = _make_group_adapter()

        output = ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(), _make_input_config(graphs)
        )

        self.assertEqual(adapter.compile_graph_group.call_count, 2)
        self.assertEqual(
            set(output.artifact_paths),
            {ARTIFACT_TEXT_DECODER, ARTIFACT_TOK_EMBEDDING},
        )
        adapter.compile_model.assert_not_called()

    def test_gives_each_component_its_own_executorch_config(self):
        """Encoders do not use RPC memory, so their IO is memory-planned."""
        graphs = {
            ARTIFACT_TEXT_DECODER: _make_decoder_graphs(),
            ARTIFACT_AUDIO_ENCODER: {GRAPH_FORWARD: _make_bundle()},
        }
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(), _make_input_config(graphs)
        )

        planning = {
            call.kwargs["artifact_key"]: call.kwargs[
                "executorch_config"
            ].memory_planning_pass
            for call in adapter.compile_graph_group.call_args_list
        }
        self.assertFalse(planning[ARTIFACT_TEXT_DECODER].alloc_graph_input)
        self.assertTrue(planning[ARTIFACT_AUDIO_ENCODER].alloc_graph_input)

    def test_logs_the_model_layout_once_per_component(self):
        """Users see which layout patterns the run applies."""
        graphs = {ARTIFACT_TEXT_DECODER: _make_decoder_graphs()}

        with self.assertLogs(
            "executorch.backends.qualcomm.genai_pipeline.compilation.pass_policy",
            level="INFO",
        ) as logs:
            ExecuTorchCompilationStrategy(_make_group_adapter()).invoke(
                make_test_context(), _make_input_config(graphs)
            )

        layout_lines = [line for line in logs.output if "ModelLayout" in line]
        self.assertEqual(len(layout_lines), 1, logs.output)
        self.assertIn(ARTIFACT_TEXT_DECODER, layout_lines[0])

    def test_passes_a_component_graphs_to_one_call(self):
        """A component's graphs share a .pte, so they go in together."""
        graphs = {ARTIFACT_TEXT_DECODER: _make_decoder_graphs()}
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(), _make_input_config(graphs)
        )

        kwargs = adapter.compile_graph_group.call_args.kwargs
        self.assertEqual(
            set(kwargs["graphs"]), {GRAPH_KV_FORWARD, GRAPH_PREFILL_FORWARD}
        )
        self.assertEqual(set(kwargs["compile_specs"]), set(kwargs["graphs"]))
        self.assertEqual(kwargs["artifact_key"], ARTIFACT_TEXT_DECODER)

    def test_names_each_artifact_file_after_its_component(self):
        """Two .pte files in one directory need distinct names."""
        graphs = {
            ARTIFACT_TEXT_DECODER: _make_decoder_graphs(),
            ARTIFACT_TOK_EMBEDDING: {TOK_EMBEDDING_GRAPH_NAMES[0]: _make_bundle()},
        }
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(), _make_input_config(graphs)
        )

        file_names = {
            call.kwargs["file_name"]
            for call in adapter.compile_graph_group.call_args_list
        }
        self.assertEqual(
            file_names,
            {
                f"{TEST_MODEL_NAME}_{ARTIFACT_TEXT_DECODER}",
                f"{TEST_MODEL_NAME}_{ARTIFACT_TOK_EMBEDDING}",
            },
        )

    def test_derives_per_component_specs_rather_than_taking_them(self):
        """Specs depend on the component and the grouping, so B derives them."""
        graphs = {ARTIFACT_TEXT_DECODER: _make_decoder_graphs()}
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(),
            _make_input_config(graphs, compile_specs=None),
        )

        specs = adapter.compile_graph_group.call_args.kwargs["compile_specs"]
        for graph_name in DECODER_GRAPH_NAMES:
            self.assertTrue(specs[graph_name])

    def test_requests_multiple_contexts_when_sharded(self):
        """num_sharding only reaches lowering through the compile specs."""
        from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
            flatbuffer_to_option,
        )
        from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC

        graphs = {ARTIFACT_TEXT_DECODER: _make_decoder_graphs()}
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(), _make_input_config(graphs, num_sharding=4)
        )

        specs = adapter.compile_graph_group.call_args.kwargs["compile_specs"]
        options = next(
            flatbuffer_to_option(spec.value)
            for spec in specs[GRAPH_KV_FORWARD]
            if spec.key == QCOM_QNN_COMPILE_SPEC
        )
        self.assertTrue(options.backend_options.htp_options.use_multi_contexts)

    def test_derives_a_pass_configuration_per_graph(self):
        """Passes depend on the graph's shapes, so B derives them too."""
        graphs = {ARTIFACT_TEXT_DECODER: _make_decoder_graphs()}
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(), _make_input_config(graphs)
        )

        kwargs = adapter.compile_graph_group.call_args.kwargs
        self.assertEqual(set(kwargs["dep_table"]), set(DECODER_GRAPH_NAMES))
        self.assertEqual(set(kwargs["passes_job"]), set(DECODER_GRAPH_NAMES))

    def test_inserts_the_splitting_pass_when_sharded(self):
        """The count that sets multi-contexts is the one that splits the graph."""
        from executorch.extension.llm.custom_ops.model_sharding import SplitGraph

        graphs = {ARTIFACT_TEXT_DECODER: _make_decoder_graphs()}
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(), _make_input_config(graphs, num_sharding=4)
        )

        kwargs = adapter.compile_graph_group.call_args.kwargs
        for graph_name in DECODER_GRAPH_NAMES:
            self.assertIn(SplitGraph, kwargs["passes_job"][graph_name])
            self.assertIn(SplitGraph, kwargs["dep_table"][graph_name])

    def test_shards_each_component_by_its_own_count(self):
        """An encoder's shard count is its own, so one scalar cannot describe it."""
        from executorch.extension.llm.custom_ops.model_sharding import SplitGraph

        graphs = {
            ARTIFACT_TEXT_DECODER: _make_decoder_graphs(),
            ARTIFACT_AUDIO_ENCODER: {GRAPH_FORWARD: _make_bundle()},
        }
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(),
            _make_input_config(
                graphs,
                num_sharding={
                    ARTIFACT_TEXT_DECODER: 1,
                    ARTIFACT_AUDIO_ENCODER: 2,
                },
            ),
        )

        by_key = {
            call.kwargs["artifact_key"]: call.kwargs
            for call in adapter.compile_graph_group.call_args_list
        }
        self.assertNotIn(
            SplitGraph, by_key[ARTIFACT_TEXT_DECODER]["passes_job"][GRAPH_KV_FORWARD]
        )
        self.assertIn(
            SplitGraph, by_key[ARTIFACT_AUDIO_ENCODER]["passes_job"][GRAPH_FORWARD]
        )

    def test_disables_both_split_and_multiple_contexts_above_the_layer_count(self):
        """Specs and passes use one clamped count, so they cannot disagree."""
        from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
            flatbuffer_to_option,
        )
        from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC
        from executorch.extension.llm.custom_ops.model_sharding import SplitGraph

        meta = {**_DECODER_META, "get_n_layers": 1}
        graphs = {ARTIFACT_TEXT_DECODER: _make_decoder_graphs(meta=meta)}
        adapter = _make_group_adapter()

        with self.assertLogs(level="WARNING"):
            ExecuTorchCompilationStrategy(adapter).invoke(
                make_test_context(), _make_input_config(graphs, num_sharding=2)
            )

        kwargs = adapter.compile_graph_group.call_args.kwargs
        options = next(
            flatbuffer_to_option(spec.value)
            for spec in kwargs["compile_specs"][GRAPH_KV_FORWARD]
            if spec.key == QCOM_QNN_COMPILE_SPEC
        )
        self.assertFalse(options.backend_options.htp_options.use_multi_contexts)
        self.assertNotIn(SplitGraph, kwargs["passes_job"][GRAPH_KV_FORWARD])

    def test_keeps_the_first_component_etrecord_and_logs_it(self):
        """One output holds one ETRecord, so the choice is logged."""
        first, second = MagicMock(name="etrecord_0"), MagicMock(name="etrecord_1")
        etrecords = {ARTIFACT_TEXT_DECODER: first, ARTIFACT_TOK_EMBEDDING: second}
        adapter = MagicMock()
        adapter.compile_graph_group.side_effect = lambda **kwargs: CompilationResult(
            artifact_paths={kwargs["artifact_key"]: Path("/tmp/x.pte")},
            etrecord=etrecords[kwargs["artifact_key"]],
        )
        graphs = {
            ARTIFACT_TEXT_DECODER: _make_decoder_graphs(),
            ARTIFACT_TOK_EMBEDDING: {TOK_EMBEDDING_GRAPH_NAMES[0]: _make_bundle()},
        }

        with self.assertLogs(level="INFO") as logs:
            output = ExecuTorchCompilationStrategy(adapter).invoke(
                make_test_context(), _make_input_config(graphs)
            )

        self.assertIs(output.etrecord, first)
        self.assertTrue(
            any(
                ARTIFACT_TEXT_DECODER in line and "ETRecord" in line
                for line in logs.output
            ),
            logs.output,
        )

    def test_tags_the_token_embedding_from_the_decoder_shapes(self):
        """The embedding has no shapes of its own, so it borrows the decoder's."""
        from executorch.backends.qualcomm._passes import TagQuantIO
        from executorch.backends.qualcomm.utils.constants import QCOM_PASS_ACTIVATE_KEY

        dtypes = {"kv_type": torch.uint8, "io_type": torch.uint16}
        graphs = {
            ARTIFACT_TEXT_DECODER: _make_decoder_graphs(quant_io_dtypes=dtypes),
            ARTIFACT_TOK_EMBEDDING: {
                TOK_EMBEDDING_GRAPH_NAMES[0]: _make_bundle(
                    meta={}, quant_io_dtypes=dtypes
                ),
            },
        }
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(), _make_input_config(graphs)
        )

        by_key = {
            call.kwargs["artifact_key"]: call.kwargs
            for call in adapter.compile_graph_group.call_args_list
        }
        embedding_passes = by_key[ARTIFACT_TOK_EMBEDDING]["passes_job"][
            TOK_EMBEDDING_GRAPH_NAMES[0]
        ]
        self.assertTrue(embedding_passes[TagQuantIO][QCOM_PASS_ACTIVATE_KEY])

    def test_compiles_a_skipped_multimodal_model_per_component_in_fp16(self):
        """With quantization skipped (GPU/FP16), MLLM still fans out."""
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_VISION_ENCODER,
        )
        from executorch.backends.qualcomm.serialization.qc_schema import (
            QnnExecuTorchHtpPrecision,
        )
        from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
            flatbuffer_to_option,
        )
        from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC

        decoder_inputs = ("tok", ("mask",), "pos", ("k",), ("v",))
        decoder_meta = {**_DECODER_META, "get_use_kv_cache": True}
        adapter = _make_group_adapter()

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(),
            CompilationInputConfig(
                soc_model=TEST_SOC_CHIPSET,
                backend_type=TEST_BACKEND_TYPE,
                model={
                    ARTIFACT_TEXT_DECODER: MagicMock(name="decoder"),
                    ARTIFACT_VISION_ENCODER: MagicMock(name="encoder"),
                },
                example_inputs={
                    ARTIFACT_TEXT_DECODER: {
                        GRAPH_FORWARD: decoder_inputs,
                        GRAPH_KV_FORWARD: decoder_inputs,
                    },
                    ARTIFACT_VISION_ENCODER: {GRAPH_FORWARD: ("pixels",)},
                },
                meta={
                    ARTIFACT_TEXT_DECODER: {
                        GRAPH_FORWARD: decoder_meta,
                        GRAPH_KV_FORWARD: decoder_meta,
                    },
                },
            ),
        )

        by_key = {
            call.kwargs["artifact_key"]: call.kwargs
            for call in adapter.compile_graph_group.call_args_list
        }
        self.assertEqual(set(by_key), {ARTIFACT_TEXT_DECODER, ARTIFACT_VISION_ENCODER})
        self.assertEqual(
            set(by_key[ARTIFACT_TEXT_DECODER]["graphs"]), {GRAPH_KV_FORWARD}
        )
        options = next(
            flatbuffer_to_option(spec.value)
            for spec in by_key[ARTIFACT_VISION_ENCODER]["compile_specs"][GRAPH_FORWARD]
            if spec.key == QCOM_QNN_COMPILE_SPEC
        )
        self.assertEqual(
            options.backend_options.htp_options.precision,
            QnnExecuTorchHtpPrecision.kHtpFp16,
        )
        adapter.compile_model.assert_not_called()

    def test_rejects_a_component_mixing_quantized_and_unquantized_graphs(self):
        """One .pte gets one precision."""
        graphs = {
            ARTIFACT_TEXT_DECODER: {
                GRAPH_KV_FORWARD: _make_bundle(),
                GRAPH_PREFILL_FORWARD: _make_bundle(quantized=False),
            }
        }

        with self.assertRaises(StageError) as ctx:
            ExecuTorchCompilationStrategy(_make_group_adapter()).invoke(
                make_test_context(), _make_input_config(graphs)
            )

        self.assertIn("mixes", str(ctx.exception))

    def test_falls_back_to_the_single_graph_path_without_graphs(self):
        """A one-graph model still compiles through compile_model."""
        adapter = MagicMock()
        adapter.compile_model.return_value = CompilationResult(
            artifact_paths={ARTIFACT_TEXT_DECODER: Path("/tmp/m.pte")},
        )

        ExecuTorchCompilationStrategy(adapter).invoke(
            make_test_context(),
            CompilationInputConfig(
                soc_model=TEST_SOC_CHIPSET,
                backend_type=TEST_BACKEND_TYPE,
                model=MagicMock(name="model"),
                example_inputs=(MagicMock(name="example_input"),),
            ),
        )

        adapter.compile_model.assert_called_once()
        adapter.compile_graph_group.assert_not_called()

    def test_rejects_an_unknown_component_key(self):
        """The outer key space is the artifact keys the runner addresses."""
        graphs = {"not_a_component": _make_decoder_graphs()}

        with self.assertRaises(StageError) as ctx:
            ExecuTorchCompilationStrategy(_make_group_adapter()).invoke(
                make_test_context(), _make_input_config(graphs)
            )

        self.assertIn("not_a_component", str(ctx.exception))

    def test_rejects_a_component_with_no_graphs(self):
        """An empty component would otherwise write an empty .pte."""
        graphs = {ARTIFACT_TEXT_DECODER: {}}

        with self.assertRaises(StageError):
            ExecuTorchCompilationStrategy(_make_group_adapter()).invoke(
                make_test_context(), _make_input_config(graphs)
            )

    def test_rejects_input_naming_nothing_to_compile(self):
        """Neither graphs nor a model means the previous stage produced nothing."""
        with self.assertRaises(StageError):
            ExecuTorchCompilationStrategy(_make_group_adapter()).invoke(
                make_test_context(),
                CompilationInputConfig(
                    soc_model=TEST_SOC_CHIPSET,
                    backend_type=TEST_BACKEND_TYPE,
                ),
            )

    def test_wraps_an_adapter_failure_in_a_stage_error(self):
        """A lowering failure must name the stage it came from."""
        adapter = MagicMock()
        adapter.compile_graph_group.side_effect = RuntimeError("lowering blew up")
        graphs = {ARTIFACT_TEXT_DECODER: _make_decoder_graphs()}

        with self.assertRaises(StageError) as ctx:
            ExecuTorchCompilationStrategy(adapter).invoke(
                make_test_context(), _make_input_config(graphs)
            )

        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)


if __name__ == "__main__":
    unittest.main()
