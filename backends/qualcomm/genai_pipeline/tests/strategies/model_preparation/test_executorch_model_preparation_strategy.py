# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import call, MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)

from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_input_config import (
    ModelPreparationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_output_config import (
    ModelPreparationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.executorch_model_preparation_strategy import (
    ExecuTorchModelPreparationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.model_preparation_strategy import (
    ModelPreparationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)

DEFAULT_GRAPH_NAME = GRAPH_FORWARD
DECODE_GRAPH_NAME = DECODER_GRAPH_NAMES[0]
TOK_EMBEDDING_GRAPH_NAME = TOK_EMBEDDING_GRAPH_NAMES[0]


def _make_modules():
    return {
        ARTIFACT_TEXT_DECODER: {
            DEFAULT_GRAPH_NAME: MagicMock(name=DEFAULT_GRAPH_NAME),
            DECODE_GRAPH_NAME: MagicMock(name=DECODE_GRAPH_NAME),
        },
        ARTIFACT_TOK_EMBEDDING: {
            DEFAULT_GRAPH_NAME: MagicMock(name=DEFAULT_GRAPH_NAME),
            TOK_EMBEDDING_GRAPH_NAME: MagicMock(name=TOK_EMBEDDING_GRAPH_NAME),
        },
        ARTIFACT_VISION_ENCODER: {
            DEFAULT_GRAPH_NAME: MagicMock(name=DEFAULT_GRAPH_NAME),
        },
    }


def _make_mock_adapter():
    """Create a mock model loader adapter with sensible defaults."""
    adapter = MagicMock()
    modules = _make_modules()
    tokenizer = MagicMock(name="tokenizer")
    # Default to "no chat template" so tests opt in explicitly.
    tokenizer.chat_template = None

    text_decoder_metadata = [
        (
            modules[ARTIFACT_TEXT_DECODER][DEFAULT_GRAPH_NAME],
            {"get_max_context_len": 1024},
        ),
        (
            modules[ARTIFACT_TEXT_DECODER][DECODE_GRAPH_NAME],
            {"get_max_context_len": 1024},
        ),
    ]

    def _get_metadata(model):
        for graph_module, metadata in text_decoder_metadata:
            if model is graph_module:
                return metadata
        return {}

    adapter.load_model.return_value = modules
    adapter.get_example_inputs.side_effect = lambda model: (model,)
    adapter.get_metadata.side_effect = _get_metadata
    adapter.apply_module_transforms.side_effect = (
        lambda module, module_transforms: module
    )
    adapter.load_tokenizer.return_value = tokenizer
    adapter.get_inference.return_value = MagicMock(name="inference")
    adapter.export_tokenizer.return_value = Path("/tmp/tokenizer/tokenizer.json")
    return adapter


def _make_strategy(loader=None):
    return ExecuTorchModelPreparationStrategy(
        model_loader_adapter=loader if loader is not None else _make_mock_adapter()
    )


def _make_valid_input_config(**overrides):
    """Create a valid ModelPreparationInputConfig with defaults."""
    defaults = {
        "model_name": "test_model",
        "soc_model": "SM8750",
    }
    defaults.update(overrides)
    return ModelPreparationInputConfig(**defaults)


class TestExecuTorchModelPreparationStrategy(unittest.TestCase):

    def test_is_model_preparation_strategy(self):
        """Strategy inherits from ModelPreparationStrategy ABC."""
        self.assertIsInstance(_make_strategy(), ModelPreparationStrategy)

    def test_default_adapters_created_when_none_provided(self):
        """Adapters fall back to their default implementations."""
        with patch(
            "executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation."
            "default_model_loader_adapter.DefaultModelLoaderAdapter"
        ) as mock_loader_cls:
            strategy = ExecuTorchModelPreparationStrategy()

            mock_loader_cls.assert_called_once()
            self.assertIs(strategy.adapter, mock_loader_cls.return_value)

    def test_custom_adapter_injected(self):
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        self.assertIs(strategy.adapter, loader)

    def test_invoke_happy_path(self):
        """Full model preparation pipeline runs successfully end-to-end."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        modules = loader.load_model.return_value
        text_decoder_forward = modules[ARTIFACT_TEXT_DECODER][DEFAULT_GRAPH_NAME]
        text_decoder_kv_forward = modules[ARTIFACT_TEXT_DECODER][DECODE_GRAPH_NAME]
        tok_embedding_forward = modules[ARTIFACT_TOK_EMBEDDING][DEFAULT_GRAPH_NAME]
        tok_embedding_kv_forward = modules[ARTIFACT_TOK_EMBEDDING][
            TOK_EMBEDDING_GRAPH_NAME
        ]
        vision_encoder_forward = modules[ARTIFACT_VISION_ENCODER][DEFAULT_GRAPH_NAME]

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        self.assertIsInstance(result, ModelPreparationOutputConfig)
        self.assertEqual(
            result.model_module,
            {
                ARTIFACT_TEXT_DECODER: text_decoder_forward,
                ARTIFACT_TOK_EMBEDDING: tok_embedding_forward,
                ARTIFACT_VISION_ENCODER: vision_encoder_forward,
            },
        )
        self.assertIs(result.tokenizer, loader.load_tokenizer.return_value)
        self.assertEqual(
            result.example_inputs,
            {
                ARTIFACT_TEXT_DECODER: {
                    DEFAULT_GRAPH_NAME: (text_decoder_forward,),
                    DECODE_GRAPH_NAME: (text_decoder_kv_forward,),
                },
                ARTIFACT_TOK_EMBEDDING: {
                    DEFAULT_GRAPH_NAME: (tok_embedding_forward,),
                    TOK_EMBEDDING_GRAPH_NAME: (tok_embedding_kv_forward,),
                },
                ARTIFACT_VISION_ENCODER: {
                    DEFAULT_GRAPH_NAME: (vision_encoder_forward,),
                },
            },
        )
        self.assertEqual(
            result.meta,
            {
                ARTIFACT_TEXT_DECODER: {
                    DEFAULT_GRAPH_NAME: {"get_max_context_len": 1024},
                    DECODE_GRAPH_NAME: {"get_max_context_len": 1024},
                }
            },
        )
        self.assertIs(result.inference, loader.get_inference.return_value)

    def test_invoke_builds_example_inputs_for_each_component_graph(self):
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        modules = loader.load_model.return_value
        expected_modules = [
            modules[ARTIFACT_TEXT_DECODER][DEFAULT_GRAPH_NAME],
            modules[ARTIFACT_TEXT_DECODER][DECODE_GRAPH_NAME],
            modules[ARTIFACT_TOK_EMBEDDING][DEFAULT_GRAPH_NAME],
            modules[ARTIFACT_TOK_EMBEDDING][TOK_EMBEDDING_GRAPH_NAME],
            modules[ARTIFACT_VISION_ENCODER][DEFAULT_GRAPH_NAME],
        ]

        strategy.invoke(make_test_context(), _make_valid_input_config())

        loader.get_example_inputs.assert_has_calls(
            [call(module) for module in expected_modules],
            any_order=False,
        )

    def test_invoke_builds_metadata_for_each_component_graph(self):
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        modules = loader.load_model.return_value
        expected_modules = [
            modules[ARTIFACT_TEXT_DECODER][DEFAULT_GRAPH_NAME],
            modules[ARTIFACT_TEXT_DECODER][DECODE_GRAPH_NAME],
            modules[ARTIFACT_TOK_EMBEDDING][DEFAULT_GRAPH_NAME],
            modules[ARTIFACT_TOK_EMBEDDING][TOK_EMBEDDING_GRAPH_NAME],
            modules[ARTIFACT_VISION_ENCODER][DEFAULT_GRAPH_NAME],
        ]

        result = strategy.invoke(make_test_context(), _make_valid_input_config())

        loader.get_metadata.assert_has_calls(
            [call(module) for module in expected_modules],
            any_order=False,
        )
        self.assertNotIn(ARTIFACT_VISION_ENCODER, result.meta)

    def test_invoke_selects_first_graph_module_before_module_transforms(self):
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        text_transform = MagicMock(name="text_transform")
        input_config = _make_valid_input_config(
            extra_options={
                "model_options": {
                    "module_transforms": {ARTIFACT_TEXT_DECODER: [text_transform]}
                }
            }
        )

        result = strategy.invoke(make_test_context(), input_config)

        loader.apply_module_transforms.assert_has_calls(
            [
                call(
                    loader.load_model.return_value[ARTIFACT_TEXT_DECODER][
                        DEFAULT_GRAPH_NAME
                    ],
                    module_transforms=[text_transform],
                ),
                call(
                    loader.load_model.return_value[ARTIFACT_TOK_EMBEDDING][
                        DEFAULT_GRAPH_NAME
                    ],
                    module_transforms=[],
                ),
                call(
                    loader.load_model.return_value[ARTIFACT_VISION_ENCODER][
                        DEFAULT_GRAPH_NAME
                    ],
                    module_transforms=[],
                ),
            ],
            any_order=False,
        )
        self.assertIs(
            result.model_module[ARTIFACT_TEXT_DECODER],
            loader.load_model.return_value[ARTIFACT_TEXT_DECODER][DEFAULT_GRAPH_NAME],
        )
        self.assertIs(
            result.model_module[ARTIFACT_TOK_EMBEDDING],
            loader.load_model.return_value[ARTIFACT_TOK_EMBEDDING][DEFAULT_GRAPH_NAME],
        )

    def test_get_component_module_does_not_filter_by_deploy_graph_name(self):
        strategy = _make_strategy()
        text_decoder_kv_forward = MagicMock(name="text_decoder_kv_forward")
        tok_embedding_kv_forward = MagicMock(name="tok_embedding_kv_forward")

        result = strategy._get_component_module(
            {
                ARTIFACT_TEXT_DECODER: {DECODE_GRAPH_NAME: text_decoder_kv_forward},
                ARTIFACT_TOK_EMBEDDING: {
                    TOK_EMBEDDING_GRAPH_NAME: tok_embedding_kv_forward
                },
                ARTIFACT_VISION_ENCODER: {},
            }
        )

        self.assertEqual(
            result,
            {
                ARTIFACT_TEXT_DECODER: text_decoder_kv_forward,
                ARTIFACT_TOK_EMBEDDING: tok_embedding_kv_forward,
            },
        )

    def test_get_component_module_does_not_mutate_graph_modules(self):
        strategy = _make_strategy()
        text_decoder_forward = MagicMock(name="text_decoder_forward")
        text_decoder_kv_forward = MagicMock(name="text_decoder_kv_forward")
        modules = {
            ARTIFACT_TEXT_DECODER: {
                DEFAULT_GRAPH_NAME: text_decoder_forward,
                DECODE_GRAPH_NAME: text_decoder_kv_forward,
            }
        }

        result = strategy._get_component_module(modules)

        self.assertEqual(result, {ARTIFACT_TEXT_DECODER: text_decoder_forward})
        self.assertEqual(
            modules,
            {
                ARTIFACT_TEXT_DECODER: {
                    DEFAULT_GRAPH_NAME: text_decoder_forward,
                    DECODE_GRAPH_NAME: text_decoder_kv_forward,
                }
            },
        )

    def test_invoke_passes_tokenizer_options_to_load_tokenizer(self):
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        tokenizer_options = {"use_fast": False}

        strategy.invoke(
            make_test_context(),
            _make_valid_input_config(
                extra_options={"tokenizer_options": tokenizer_options}
            ),
        )

        loader.load_tokenizer.assert_called_once_with(
            model_name="test_model",
            extra_options=tokenizer_options,
        )

    def test_invoke_builds_inference_from_meta_and_example_inputs(self):
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        extra = {"embedding_quantize": "4a"}

        result = strategy.invoke(
            make_test_context(), _make_valid_input_config(extra_options=extra)
        )

        loader.get_inference.assert_called_once_with(
            result.meta,
            result.example_inputs,
            extra_options=extra,
        )

    def test_invoke_rejects_unkeyed_module_transforms(self):
        strategy = _make_strategy()
        input_config = _make_valid_input_config(
            extra_options={"model_options": {"module_transforms": []}}
        )

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), input_config)

        self.assertEqual(cm.exception.stage_name, "model_preparation")
        self.assertIsInstance(cm.exception.original_exception, ValueError)
        self.assertIn("module_transforms", str(cm.exception.original_exception))

    def test_invoke_exports_tokenizer_when_requested(self):
        """export_tokenizer is called when export_tokenizer=True in extra_options."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        input_config = _make_valid_input_config(
            extra_options={"export_tokenizer": True}
        )
        context = make_test_context(artifact_dir="/my/artifacts")

        result = strategy.invoke(context, input_config)

        loader.export_tokenizer.assert_called_once_with(
            tokenizer=loader.load_tokenizer.return_value,
            output_dir=Path("/my/artifacts") / "tokenizer",
            extra_options=None,
        )
        # The adapter returns the tokenizer file itself, not its directory.
        self.assertEqual(
            result.runtime_tokenizer_path, Path("/tmp/tokenizer/tokenizer.json")
        )

    def test_invoke_chat_template_prefers_tokenizer_over_extra_options(self):
        """A chat template on the tokenizer is carried into the output config."""
        loader = _make_mock_adapter()
        loader.load_tokenizer.return_value.chat_template = "tokenizer_template"
        strategy = _make_strategy(loader)
        input_config = _make_valid_input_config(
            extra_options={"chat_template": "fallback"}
        )

        result = strategy.invoke(make_test_context(), input_config)

        self.assertEqual(result.chat_template, "tokenizer_template")

    def test_invoke_missing_model_name_raises_stage_error(self):
        """StageError raised when model_name is empty."""
        strategy = _make_strategy()

        with self.assertRaises(StageError) as cm:
            strategy.invoke(
                make_test_context(), _make_valid_input_config(model_name="")
            )
        self.assertIn("model_name", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "model_preparation")

    def test_invoke_missing_soc_model_raises_stage_error(self):
        """StageError raised when soc_model is empty."""
        strategy = _make_strategy()

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config(soc_model=""))
        self.assertIn("soc_model", str(cm.exception))
        self.assertEqual(cm.exception.stage_name, "model_preparation")

    def test_invoke_adapter_exception_wrapped_in_stage_error(self):
        """Exceptions from the adapter are wrapped in StageError."""
        loader = _make_mock_adapter()
        loader.load_model.side_effect = RuntimeError("model load failed")
        strategy = _make_strategy(loader)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertEqual(cm.exception.stage_name, "model_preparation")
        self.assertIsInstance(cm.exception.original_exception, RuntimeError)
        self.assertIn("model load failed", str(cm.exception))

    def test_invoke_stage_error_not_double_wrapped(self):
        loader = _make_mock_adapter()
        original_error = StageError(
            stage_name="model_preparation", message="inner error"
        )
        loader.load_model.side_effect = original_error
        strategy = _make_strategy(loader)

        with self.assertRaises(StageError) as cm:
            strategy.invoke(make_test_context(), _make_valid_input_config())
        self.assertIs(cm.exception, original_error)

    def test_invoke_model_options_forwarded(self):
        """model_options from extra_options are forwarded to load_model."""
        loader = _make_mock_adapter()
        strategy = _make_strategy(loader)
        model_opts = {"torch_dtype": "float16"}
        input_config = _make_valid_input_config(
            extra_options={"model_options": model_opts}
        )

        strategy.invoke(make_test_context(), input_config)

        loader.load_model.assert_called_once_with(
            model_name="test_model",
            extra_options=model_opts,
        )


if __name__ == "__main__":
    unittest.main()
