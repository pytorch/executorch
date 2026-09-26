# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import call, MagicMock, patch

import torch
from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)

from executorch.backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.executorch_quantization_strategy import (
    ExecuTorchQuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantizer_adapter import (
    QuantizerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)

DECODE_GRAPH_NAME = DECODER_GRAPH_NAMES[0]
PREFILL_GRAPH_NAME = DECODER_GRAPH_NAMES[1]
EMBED_DECODE_GRAPH_NAME = TOK_EMBEDDING_GRAPH_NAMES[0]


def _make_recipe(name, kv_bit_width=8, logits_bit_width=16):
    recipe = MagicMock(name=name)
    recipe.recipe = {name: "recipe"}
    recipe.get_kv_io_bit_width.return_value = kv_bit_width
    recipe.get_logits_output_bit_width.return_value = logits_bit_width
    return recipe


def _make_mock_adapter():
    """Create a mock quantizer adapter with per-step return values."""
    adapter = MagicMock(spec=QuantizerAdapter)
    adapter.export_model.side_effect = lambda module, example_inputs: MagicMock(
        name=f"exported_{module._mock_name}_{example_inputs[0]}"
    )
    adapter.prepare_pt2e.side_effect = lambda module, quantizer: MagicMock(
        name=f"prepared_{module._mock_name}"
    )
    adapter.convert_pt2e.side_effect = lambda module: MagicMock(
        name=f"converted_{module._mock_name}"
    )
    return adapter


def _make_mock_calibration_adapter():
    """Create a mock calibration data adapter with component-keyed data."""
    adapter = MagicMock()
    adapter.generate_calibration_data.return_value = {
        ARTIFACT_TEXT_DECODER: MagicMock(name="text_calibration_data")
    }
    return adapter


def _make_strategy(adapter=None, calibration=None):
    """Build the strategy with quantization and calibration adapters mocked."""
    return ExecuTorchQuantizationStrategy(
        quantizer_adapter=adapter if adapter is not None else _make_mock_adapter(),
        calibration_data_adapter=(
            calibration if calibration is not None else _make_mock_calibration_adapter()
        ),
    )


def _make_valid_input_config(**overrides):
    """Create a valid QuantizationInputConfig with component/graph defaults."""
    defaults = {
        "soc_model": MagicMock(name="SM8750"),
        "backend_type": MagicMock(name="kHtpBackend"),
        "model_module": {
            ARTIFACT_TEXT_DECODER: MagicMock(name="decoder"),
            ARTIFACT_TOK_EMBEDDING: MagicMock(name="tok_embedding"),
            ARTIFACT_VISION_ENCODER: MagicMock(name="vision_encoder"),
        },
        "example_inputs": {
            ARTIFACT_TEXT_DECODER: {
                GRAPH_FORWARD: (
                    "text_calib",
                    ("text_calib_mask",),
                    "text_calib_pos_ids",
                    ("text_calib_k_cache",),
                    ("text_calib_v_cache",),
                ),
                DECODE_GRAPH_NAME: (
                    "text_decode",
                    ("text_decode_mask",),
                    "text_decode_pos_ids",
                    ("text_decode_k_cache",),
                    ("text_decode_v_cache",),
                ),
                PREFILL_GRAPH_NAME: (
                    "text_prefill",
                    ("text_prefill_mask",),
                    "text_prefill_pos_ids",
                    ("text_prefill_k_cache",),
                    ("text_prefill_v_cache",),
                ),
            },
            ARTIFACT_TOK_EMBEDDING: {
                GRAPH_FORWARD: ("embed_calib",),
                EMBED_DECODE_GRAPH_NAME: ("embed_decode",),
            },
            ARTIFACT_VISION_ENCODER: {
                GRAPH_FORWARD: ("vision",),
            },
        },
        "tokenizer": MagicMock(name="tokenizer"),
        "meta": {
            ARTIFACT_TEXT_DECODER: {
                GRAPH_FORWARD: {
                    "get_n_layers": 2,
                    "get_use_kv_cache": False,
                },
                DECODE_GRAPH_NAME: {"get_n_layers": 2, "get_use_kv_cache": True},
                PREFILL_GRAPH_NAME: {"get_n_layers": 2, "get_use_kv_cache": False},
            },
            ARTIFACT_TOK_EMBEDDING: {
                GRAPH_FORWARD: {"get_n_layers": 0},
                EMBED_DECODE_GRAPH_NAME: {"get_n_layers": 0},
            },
            ARTIFACT_VISION_ENCODER: {
                GRAPH_FORWARD: {},
            },
        },
        "inference": MagicMock(name="inference"),
        "extra_options": {
            "quantize_options": {
                "quant_recipe": {
                    ARTIFACT_TEXT_DECODER: _make_recipe("text_recipe"),
                    ARTIFACT_TOK_EMBEDDING: None,
                    ARTIFACT_VISION_ENCODER: _make_recipe("vision_recipe"),
                }
            }
        },
    }
    defaults.update(overrides)
    return QuantizationInputConfig(**defaults)


class TestExecuTorchQuantizationStrategy(unittest.TestCase):
    def setUp(self):
        self.save_quantized_module = patch(
            "executorch.backends.qualcomm.genai_pipeline.quant_utilities."
            "save_quantized_module"
        ).start()
        self.encoding_override = patch(
            "executorch.backends.qualcomm.genai_pipeline.quant_utilities."
            "encoding_override"
        ).start()
        self.save_logits_quant_attrs = patch(
            "executorch.backends.qualcomm.genai_pipeline.quant_utilities."
            "save_logits_quant_attrs"
        ).start()
        self.save_output_kv_cache_quant_attrs = patch(
            "executorch.backends.qualcomm.genai_pipeline.quant_utilities."
            "save_output_kv_cache_quant_attrs"
        ).start()
        self.make_quantizer = patch(
            "executorch.backends.qualcomm.genai_pipeline.quant_utilities."
            "make_quantizer"
        ).start()
        self.addCleanup(patch.stopall)

    def test_is_quantization_strategy(self):
        """Strategy inherits from QuantizationStrategy ABC."""
        strategy = _make_strategy()
        self.assertIsInstance(strategy, QuantizationStrategy)

    def test_default_adapters_created_when_none_provided(self):
        """Default quantization and purpose adapters are created when omitted."""
        with patch(
            "executorch.backends.qualcomm.genai_pipeline.strategies.quantization."
            "default_quantizer_adapter.DefaultQuantizerAdapter"
        ) as quantizer_cls, patch(
            "executorch.backends.qualcomm.genai_pipeline.datasets.calibration."
            "default_calibration_data_adapter.DefaultCalibrationDataAdapter"
        ) as calibration_cls, patch(
            "executorch.backends.qualcomm.genai_pipeline.datasets.training."
            "default_training_data_adapter.DefaultTrainingDataAdapter"
        ) as training_cls, patch(
            "executorch.backends.qualcomm.genai_pipeline.datasets.evaluation."
            "default_evaluation_data_adapter.DefaultEvaluationDataAdapter"
        ) as evaluation_cls:
            strategy = ExecuTorchQuantizationStrategy()

            quantizer_cls.assert_called_once()
            calibration_cls.assert_called_once()
            training_cls.assert_called_once()
            evaluation_cls.assert_called_once()
            self.assertIs(strategy.adapter, quantizer_cls.return_value)

    def test_custom_adapters_injected(self):
        """Custom quantizer and calibration adapters are used when provided."""
        adapter = _make_mock_adapter()
        calibration = _make_mock_calibration_adapter()
        strategy = _make_strategy(adapter, calibration)
        self.assertIs(strategy.adapter, adapter)

    def test_invoke_returns_graph_bundles_for_deployable_graphs(self):
        """Full quantization returns deployable graph bundles per component."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()

        result = strategy.invoke(make_test_context(), input_config)

        self.assertIsInstance(result, QuantizationOutputConfig)
        self.assertNotIn(GRAPH_FORWARD, result.graphs[ARTIFACT_TEXT_DECODER])
        self.assertIn(DECODE_GRAPH_NAME, result.graphs[ARTIFACT_TEXT_DECODER])
        self.assertIn(PREFILL_GRAPH_NAME, result.graphs[ARTIFACT_TEXT_DECODER])
        self.assertNotIn(GRAPH_FORWARD, result.graphs[ARTIFACT_TOK_EMBEDDING])
        self.assertIn(EMBED_DECODE_GRAPH_NAME, result.graphs[ARTIFACT_TOK_EMBEDDING])
        self.assertIn(GRAPH_FORWARD, result.graphs[ARTIFACT_VISION_ENCODER])
        self.assertIsInstance(
            result.graphs[ARTIFACT_TEXT_DECODER][DECODE_GRAPH_NAME], GraphBundle
        )

    def test_invoke_exports_each_graph_with_component_weight_holder(self):
        """export_model receives each graph's inputs and shared component module.

        The export signature comes from model preparation's example inputs
        (zero-initialized KV caches, fixed AR length), not from a calibration
        sample, which lacks the full export signature.
        """
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()

        strategy.invoke(make_test_context(), input_config)

        adapter.export_model.assert_has_calls(
            [
                call(
                    input_config.model_module[ARTIFACT_TEXT_DECODER],
                    ("text_calib", "text_calib_mask"),
                ),
                call(
                    input_config.model_module[ARTIFACT_TEXT_DECODER],
                    (
                        "text_decode",
                        "text_decode_mask",
                        "text_decode_pos_ids",
                        "text_decode_k_cache",
                        "text_decode_v_cache",
                    ),
                ),
                call(
                    input_config.model_module[ARTIFACT_TEXT_DECODER],
                    ("text_prefill", "text_prefill_mask"),
                ),
                call(
                    input_config.model_module[ARTIFACT_TOK_EMBEDDING],
                    input_config.example_inputs[ARTIFACT_TOK_EMBEDDING][GRAPH_FORWARD],
                ),
                call(
                    input_config.model_module[ARTIFACT_TOK_EMBEDDING],
                    input_config.example_inputs[ARTIFACT_TOK_EMBEDDING][
                        EMBED_DECODE_GRAPH_NAME
                    ],
                ),
                call(
                    input_config.model_module[ARTIFACT_VISION_ENCODER],
                    input_config.example_inputs[ARTIFACT_VISION_ENCODER][GRAPH_FORWARD],
                ),
            ],
            any_order=False,
        )

    def test_export_and_prepare_flattens_text_decoder_inputs(self):
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        decoder = MagicMock(name="decoder")
        tokens = MagicMock(name="tokens")
        attention_masks = (MagicMock(name="causal_mask"), MagicMock(name="local_mask"))
        pos_ids = MagicMock(name="pos_ids")
        k_caches = (MagicMock(name="k_cache_0"), MagicMock(name="k_cache_1"))
        v_caches = (MagicMock(name="v_cache_0"), MagicMock(name="v_cache_1"))
        input_config = _make_valid_input_config(
            model_module={ARTIFACT_TEXT_DECODER: decoder},
            example_inputs={
                ARTIFACT_TEXT_DECODER: {
                    DECODE_GRAPH_NAME: (
                        tokens,
                        attention_masks,
                        pos_ids,
                        k_caches,
                        v_caches,
                    )
                }
            },
            meta={
                ARTIFACT_TEXT_DECODER: {DECODE_GRAPH_NAME: {"get_use_kv_cache": True}}
            },
        )

        strategy._export_and_prepare(
            input_config,
            {ARTIFACT_TEXT_DECODER: {DECODE_GRAPH_NAME: MagicMock(name="quantizer")}},
        )

        adapter.export_model.assert_called_once_with(
            decoder,
            (tokens, *attention_masks, pos_ids, *k_caches, *v_caches),
        )

    def test_invoke_generates_calibration_data_from_adapter(self):
        """Calibration data is generated with flattened example inputs.

        The calibration adapter receives only the non-deployed graph inputs for
        each component, so collators derive masks from the calibration signature
        rather than deployed decode/prefill signatures.
        """
        calibration = _make_mock_calibration_adapter()
        strategy = _make_strategy(calibration=calibration)
        extra_options = {
            "model_options": {"model_arch": MagicMock(name="model_arch")},
            "quantize_options": {
                "quant_dtype": {
                    ARTIFACT_TEXT_DECODER: "text_dtype",
                    ARTIFACT_TOK_EMBEDDING: "embed_dtype",
                    ARTIFACT_VISION_ENCODER: "vision_dtype",
                }
            },
            "dataset_options": MagicMock(name="dataset_options"),
        }
        input_config = _make_valid_input_config(extra_options=extra_options)

        strategy.invoke(make_test_context(), input_config)

        _, kwargs = calibration.generate_calibration_data.call_args
        self.assertIs(kwargs["tokenizer"], input_config.tokenizer)
        self.assertEqual(
            kwargs["example_inputs"],
            {
                ARTIFACT_TEXT_DECODER: input_config.example_inputs[
                    ARTIFACT_TEXT_DECODER
                ][GRAPH_FORWARD],
                ARTIFACT_TOK_EMBEDDING: input_config.example_inputs[
                    ARTIFACT_TOK_EMBEDDING
                ][GRAPH_FORWARD],
                ARTIFACT_VISION_ENCODER: input_config.example_inputs[
                    ARTIFACT_VISION_ENCODER
                ][GRAPH_FORWARD],
            },
        )
        self.assertNotIn("extra_options", kwargs)

    def test_invoke_calibrates_only_non_deployed_graphs(self):
        """Only each component's non-deployed graph is truly calibrated."""
        adapter = _make_mock_adapter()
        calibration = _make_mock_calibration_adapter()
        strategy = _make_strategy(adapter, calibration)
        input_config = _make_valid_input_config()

        strategy.invoke(make_test_context(), input_config)

        quantization_graphs = adapter.calibrate.call_args[0][0]
        self.assertEqual(
            set(quantization_graphs),
            {ARTIFACT_TEXT_DECODER, ARTIFACT_TOK_EMBEDDING, ARTIFACT_VISION_ENCODER},
        )
        adapter.calibrate.assert_called_once_with(
            quantization_graphs,
            calibration.generate_calibration_data.return_value,
            inference=input_config.inference,
        )

    def test_invoke_initializes_deployed_graph_observers(self):
        """Deployed graph observers are initialized with their own inputs."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()

        strategy.invoke(make_test_context(), input_config)

        initialized_inputs = [
            args[0][1] for args in adapter.init_encodings.call_args_list
        ]
        self.assertEqual(
            initialized_inputs,
            [
                (
                    "text_decode",
                    "text_decode_mask",
                    "text_decode_pos_ids",
                    "text_decode_k_cache",
                    "text_decode_v_cache",
                ),
                ("text_prefill", "text_prefill_mask"),
                input_config.example_inputs[ARTIFACT_TOK_EMBEDDING][
                    EMBED_DECODE_GRAPH_NAME
                ],
            ],
        )

    def test_invoke_overrides_only_text_and_embedding_deployed_graphs(self):
        """Encoding override applies only where calibration/deploy graph split exists."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()

        strategy.invoke(make_test_context(), input_config)

        self.assertEqual(self.encoding_override.call_count, 3)
        n_cache_layers = [
            kwargs.get("n_cache_layers")
            for _, kwargs in self.encoding_override.call_args_list
        ]
        self.assertEqual(
            n_cache_layers,
            [
                2,
                2,
                None,
            ],
        )
        text_override_calls = self.encoding_override.call_args_list[:2]
        self.assertIs(
            text_override_calls[1].kwargs["quantized_model"],
            text_override_calls[0].kwargs["quantized_model"],
        )
        self.assertEqual(self.save_logits_quant_attrs.call_count, 2)
        self.assertEqual(self.save_output_kv_cache_quant_attrs.call_count, 2)

    def test_invoke_skips_missing_text_decoder_prefill_graph(self):
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()
        input_config.example_inputs[ARTIFACT_TEXT_DECODER].pop(PREFILL_GRAPH_NAME)
        input_config.meta[ARTIFACT_TEXT_DECODER].pop(PREFILL_GRAPH_NAME)

        strategy.invoke(make_test_context(), input_config)

        self.assertEqual(self.encoding_override.call_count, 2)
        self.assertEqual(self.save_logits_quant_attrs.call_count, 1)
        self.assertEqual(self.save_output_kv_cache_quant_attrs.call_count, 1)

    def test_invoke_saves_text_decoder_qdq_module(self):
        """The text decoder calibration graph is saved for SQNR evaluation."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()
        context = make_test_context(artifact_dir="/tmp/qdq_artifacts")

        strategy.invoke(context, input_config)

        self.save_quantized_module.assert_called_once()
        _, kwargs = self.save_quantized_module.call_args
        self.assertIs(
            kwargs["quantized_module"],
            self.encoding_override.call_args_list[0].kwargs["quantized_model"],
        )
        self.assertEqual(
            kwargs["example_inputs"],
            ("text_calib", "text_calib_mask"),
        )
        self.assertEqual(kwargs["artifact_dir"], context.artifact_dir)

    def test_save_quantized_module_flattens_quantization_graph_inputs(self):
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()
        tokens = MagicMock(name="tokens")
        attention_masks = (MagicMock(name="causal_mask"), MagicMock(name="local_mask"))
        pos_ids = MagicMock(name="pos_ids")
        k_caches = (MagicMock(name="k_cache_0"), MagicMock(name="k_cache_1"))
        v_caches = (MagicMock(name="v_cache_0"), MagicMock(name="v_cache_1"))
        input_config.example_inputs[ARTIFACT_TEXT_DECODER][GRAPH_FORWARD] = (
            tokens,
            attention_masks,
            pos_ids,
            k_caches,
            v_caches,
        )
        input_config.meta[ARTIFACT_TEXT_DECODER][GRAPH_FORWARD] = {
            "get_use_kv_cache": True
        }

        strategy.invoke(make_test_context(), input_config)

        _, kwargs = self.save_quantized_module.call_args
        self.assertEqual(
            kwargs["example_inputs"],
            (tokens, *attention_masks, pos_ids, *k_caches, *v_caches),
        )

    def test_invoke_routes_component_quant_options_to_each_graph_quantizer(self):
        """make_quantizer receives each graph's component recipe and dtype."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        text_recipe = _make_recipe("text_recipe")
        embed_recipe = _make_recipe("embed_recipe")
        vision_recipe = _make_recipe("vision_recipe")
        input_config = _make_valid_input_config(
            extra_options={
                "model_arch": MagicMock(name="model_arch"),
                "model_options": {"model_arch": MagicMock(name="nested_model_arch")},
                "quantize_options": {
                    "quant_dtype": {
                        ARTIFACT_TEXT_DECODER: "text_dtype",
                        ARTIFACT_TOK_EMBEDDING: "embed_dtype",
                        ARTIFACT_VISION_ENCODER: "vision_dtype",
                    },
                    "quant_recipe": {
                        ARTIFACT_TEXT_DECODER: text_recipe,
                        ARTIFACT_TOK_EMBEDDING: embed_recipe,
                        ARTIFACT_VISION_ENCODER: vision_recipe,
                    },
                    "per_channel_conv": True,
                },
                "dataset_options": {"max_context_len": 128},
            }
        )

        strategy.invoke(make_test_context(), input_config)

        self.make_quantizer.assert_has_calls(
            [
                call(
                    backend=input_config.backend_type,
                    soc_model=input_config.soc_model,
                    quant_dtype="text_dtype",
                    quant_recipe=text_recipe,
                ),
                call(
                    backend=input_config.backend_type,
                    soc_model=input_config.soc_model,
                    quant_dtype="text_dtype",
                    quant_recipe=text_recipe,
                ),
                call(
                    backend=input_config.backend_type,
                    soc_model=input_config.soc_model,
                    quant_dtype="text_dtype",
                    quant_recipe=text_recipe,
                ),
                call(
                    backend=input_config.backend_type,
                    soc_model=input_config.soc_model,
                    quant_dtype="embed_dtype",
                    quant_recipe=embed_recipe,
                ),
                call(
                    backend=input_config.backend_type,
                    soc_model=input_config.soc_model,
                    quant_dtype="embed_dtype",
                    quant_recipe=embed_recipe,
                ),
                call(
                    backend=input_config.backend_type,
                    soc_model=input_config.soc_model,
                    quant_dtype="vision_dtype",
                    quant_recipe=vision_recipe,
                ),
            ],
            any_order=False,
        )

    def test_make_quantizer_sets_recipe_verbose_only_for_calibration_graph(self):
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()
        verbose_values = []

        class RecordingRecipe:
            def __init__(self, *, verbose):
                verbose_values.append(verbose)

        quantizers, _ = strategy._make_quantizer(
            input_config,
            {
                "quant_recipe": {ARTIFACT_TEXT_DECODER: RecordingRecipe},
            },
        )

        self.assertEqual(
            set(quantizers[ARTIFACT_TEXT_DECODER]),
            {GRAPH_FORWARD, DECODE_GRAPH_NAME, PREFILL_GRAPH_NAME},
        )
        self.assertEqual(verbose_values, [True, False, False])

    def test_invoke_omits_quant_dtype_when_not_set(self):
        """quant_dtype is omitted when not explicitly set in extra_options."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()

        strategy.invoke(make_test_context(), input_config)

        for _, kwargs in self.make_quantizer.call_args_list:
            self.assertNotIn("quant_dtype", kwargs)

    def test_invoke_populates_graph_bundle_fields(self):
        """GraphBundle carries the converted module, inputs, metadata and IO dtype."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config()

        result = strategy.invoke(make_test_context(), input_config)

        bundle = result.graphs[ARTIFACT_TEXT_DECODER][DECODE_GRAPH_NAME]
        self.assertIs(
            bundle.inputs,
            input_config.example_inputs[ARTIFACT_TEXT_DECODER][DECODE_GRAPH_NAME],
        )
        self.assertIs(
            bundle.meta, input_config.meta[ARTIFACT_TEXT_DECODER][DECODE_GRAPH_NAME]
        )
        self.assertEqual(
            bundle.quant_io_dtypes,
            {"kv_type": torch.uint8, "io_type": torch.uint16},
        )

    def test_quant_io_dtypes_returns_none_for_missing_recipe(self):
        """Components without an IO recipe leave GraphBundle IO dtypes unset."""
        strategy = _make_strategy()

        self.assertIsNone(strategy._get_quant_io_dtypes(None))

    def test_quant_io_dtypes_rejects_unsupported_widths(self):
        """Unsupported IO widths must not silently produce a partial result."""
        strategy = _make_strategy()

        with self.assertRaisesRegex(RuntimeError, "Unsupported quantization IO"):
            strategy._get_quant_io_dtypes(
                _make_recipe("unquantized_recipe", kv_bit_width=32)
            )

    def test_invoke_rejects_unsupported_quant_io_widths(self):
        """An invalid recipe must surface as a quantization failure."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config(
            extra_options={
                "quantize_options": {
                    "quant_recipe": {
                        ARTIFACT_TEXT_DECODER: _make_recipe(
                            "text_recipe",
                            kv_bit_width=32,
                            logits_bit_width=32,
                        ),
                    }
                }
            }
        )

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)
        self.assertIn("Unsupported quantization IO", str(cm.exception))

    def test_invoke_allows_components_without_metadata(self):
        """Graph bundles tolerate modules that do not expose get_metadata()."""
        adapter = _make_mock_adapter()
        strategy = _make_strategy(adapter)
        input_config = _make_valid_input_config(
            meta={
                ARTIFACT_TEXT_DECODER: {
                    GRAPH_FORWARD: {
                        "get_n_layers": 2,
                        "get_use_kv_cache": False,
                    },
                    DECODE_GRAPH_NAME: {
                        "get_n_layers": 2,
                        "get_use_kv_cache": True,
                    },
                    PREFILL_GRAPH_NAME: {
                        "get_n_layers": 2,
                        "get_use_kv_cache": False,
                    },
                }
            }
        )

        result = strategy.invoke(make_test_context(), input_config)

        self.assertEqual(
            result.graphs[ARTIFACT_TOK_EMBEDDING][EMBED_DECODE_GRAPH_NAME].meta,
            {},
        )

    def test_invoke_missing_example_inputs_raises_stage_error(self):
        """StageError raised when example_inputs is None."""
        strategy = _make_strategy()
        input_config = _make_valid_input_config(example_inputs=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)

        self.assertIn("example_inputs", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "quantization")

    def test_invoke_missing_model_raises_stage_error(self):
        """StageError raised when model_module is None."""
        strategy = _make_strategy()
        input_config = _make_valid_input_config(model_module=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)

        self.assertIn("model_module", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "quantization")

    def test_invoke_missing_backend_type_raises_stage_error(self):
        """StageError raised when backend_type is None."""
        strategy = _make_strategy()
        input_config = _make_valid_input_config(backend_type=None)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)

        self.assertIn("backend_type", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "quantization")

    def test_invoke_rejects_per_component_backend_map(self):
        """backend_type must be one shared QNN backend, not a component map."""
        strategy = _make_strategy()
        input_config = _make_valid_input_config(
            backend_type={ARTIFACT_TEXT_DECODER: "htp"}
        )

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)

        self.assertIn("backend_type must be one shared QNN backend", str(cm.exception))

    def test_invoke_adapter_exception_wrapped_in_stage_error(self):
        """Exceptions from the adapter are wrapped in StageError."""
        adapter = _make_mock_adapter()
        adapter.export_model.side_effect = RuntimeError("export failed")
        strategy = _make_strategy(adapter)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertEqual(cm.exception.stage_name, "quantization")
        self.assertIsInstance(cm.exception.original_exception, RuntimeError)
        self.assertIn("export failed", str(cm.exception))

    def test_invoke_stage_error_not_double_wrapped(self):
        """StageError from adapter is re-raised directly, not wrapped again."""
        adapter = _make_mock_adapter()
        original_error = StageError(stage_name="quantization", message="inner error")
        adapter.export_model.side_effect = original_error
        strategy = _make_strategy(adapter)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertIs(cm.exception, original_error)


if __name__ == "__main__":
    unittest.main()
