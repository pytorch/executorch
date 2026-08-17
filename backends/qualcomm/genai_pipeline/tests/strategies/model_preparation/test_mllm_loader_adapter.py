# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)

from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.mllm_loader_adapter import (
    MLLMLoaderAdapter,
)

DEFAULT_GRAPH_NAME = GRAPH_FORWARD
DECODE_GRAPH_NAME = DECODER_GRAPH_NAMES[0]
EMBED_DECODE_GRAPH_NAME = TOK_EMBEDDING_GRAPH_NAMES[0]
TEST_TOKENIZER_CONFIG = "tokenizer_config.json"
TEST_SPECIAL_TOKENS_MAP = "special_tokens_map.json"
TEST_TOKENIZER_JSON = "tokenizer.json"
TEST_TOKENIZER_MODEL = "tokenizer.model"
TEST_ADDED_TOKENS = "added_tokens.json"


def _make_model_config():
    model_config = MagicMock()
    model_config.repo_id = "repo"
    model_config.convert_weights = MagicMock()
    vision_config = MagicMock()
    vision_config.create_encoder.return_value = MagicMock(name="encoder")
    vision_config.create_encoder.return_value.eval.return_value = (
        vision_config.create_encoder.return_value
    )
    model_config.vision_encoder = MagicMock(return_value=vision_config)
    return model_config


def _make_adapter():
    control_args = MagicMock()
    control_args.checkpoint = "/tmp/checkpoint.pt"
    control_args.model = "test_model"
    return MLLMLoaderAdapter(_make_model_config(), control_args)


def _make_module(name):
    module = MagicMock(name=name)
    module.eval.return_value = module
    return module


class TestMLLMLoaderAdapter(unittest.TestCase):
    def test_requires_modality_encoder_config(self):
        model_config = MagicMock(spec=[])

        with self.assertRaises(ValueError):
            MLLMLoaderAdapter(model_config, MagicMock())

    def test_get_metadata_reads_module_when_it_exposes_metadata(self):
        adapter = _make_adapter()
        decoder = MagicMock()
        decoder.get_metadata.return_value = {"get_n_layers": 2}

        result = adapter.get_metadata(decoder)

        self.assertEqual(result, {"get_n_layers": 2})
        decoder.get_metadata.assert_called_once_with()

    def test_get_metadata_returns_empty_when_module_exposes_none(self):
        adapter = _make_adapter()
        vision = MagicMock(spec=[])

        result = adapter.get_metadata(vision)

        self.assertEqual(result, {})

    def test_load_embedding_builds_each_graph_with_shared_weights(self):
        adapter = _make_adapter()
        weights = MagicMock(name="weights")
        weights.to.return_value = weights
        auto_model = MagicMock()
        auto_model.get_input_embeddings.return_value = weights
        build_calib = MagicMock(return_value=MagicMock(name="embed_calib"))
        build_decode = MagicMock(return_value=MagicMock(name="embed_decode"))

        result = adapter._load_embedding(
            auto_model,
            {
                DEFAULT_GRAPH_NAME: build_calib,
                EMBED_DECODE_GRAPH_NAME: build_decode,
            },
        )

        build_calib.assert_called_once_with(weights)
        build_decode.assert_called_once_with(weights)
        self.assertEqual(
            result,
            {
                DEFAULT_GRAPH_NAME: build_calib.return_value,
                EMBED_DECODE_GRAPH_NAME: build_decode.return_value,
            },
        )

    def test_load_encoder_wraps_shared_auto_model_state(self):
        adapter = _make_adapter()
        auto_model = MagicMock()
        auto_model.state_dict.return_value = {"encoder.weight": MagicMock()}
        encoder = (
            adapter.model_config.vision_encoder.return_value.create_encoder.return_value
        )

        result = adapter._load_encoder(ARTIFACT_VISION_ENCODER, auto_model)

        adapter.model_config.vision_encoder.assert_called_once_with()
        adapter.model_config.vision_encoder.return_value.create_encoder.assert_called_once_with(
            auto_model.config
        )
        encoder.load_state_dict.assert_called_once_with(
            auto_model.state_dict.return_value, strict=False
        )
        self.assertEqual(result, {DEFAULT_GRAPH_NAME: encoder})

    def test_load_decoder_loads_checkpoint_into_each_graph(self):
        adapter = _make_adapter()
        calibration_module = _make_module("decoder_calibration")
        decode_module = _make_module("decoder_decode")
        state_dict = {"weight": MagicMock(name="weight")}
        transform = MagicMock(return_value={"weight": MagicMock(name="rewritten")})

        with patch("torch.load", return_value=state_dict) as torch_load:
            result = adapter._load_decoder(
                {
                    DEFAULT_GRAPH_NAME: MagicMock(return_value=calibration_module),
                    DECODE_GRAPH_NAME: MagicMock(return_value=decode_module),
                },
                [transform],
            )

        torch_load.assert_called_once_with(
            "/tmp/checkpoint.pt", weights_only=True, map_location="cpu", mmap=True
        )
        transform.assert_called_once_with(state_dict)
        calibration_module.load_state_dict.assert_called_once_with(
            transform.return_value, strict=True, assign=True
        )
        decode_module.load_state_dict.assert_called_once_with(
            transform.return_value, strict=True, assign=True
        )
        self.assertEqual(
            result,
            {
                DEFAULT_GRAPH_NAME: calibration_module,
                DECODE_GRAPH_NAME: decode_module,
            },
        )

    def test_load_decoder_state_dict_uses_injected_hf_loader(self):
        adapter = _make_adapter()
        state_dict = {"weight": MagicMock(name="weight")}
        state_dict_loader = MagicMock(return_value=state_dict)
        adapter.control_args.checkpoint = None
        adapter.model_config.repo_id = "test/repo"

        with patch("torch.load") as torch_load:
            result = adapter._load_decoder_state_dict(state_dict_loader)

        state_dict_loader.assert_called_once_with("test/repo")
        torch_load.assert_not_called()
        self.assertIs(result, state_dict)

    def test_load_decoder_passes_injected_hf_loader(self):
        adapter = _make_adapter()
        calibration_module = _make_module("decoder_calibration")
        state_dict_loader = MagicMock(name="state_dict_loader")
        state_dict = {"weight": MagicMock(name="weight")}

        with patch.object(
            adapter,
            "_load_decoder_state_dict",
            return_value=state_dict,
        ) as load_state_dict:
            adapter._load_decoder(
                {
                    DEFAULT_GRAPH_NAME: MagicMock(return_value=calibration_module),
                },
                [],
                state_dict_loader,
            )

        load_state_dict.assert_called_once_with(state_dict_loader)
        calibration_module.load_state_dict.assert_called_once_with(
            state_dict, strict=True, assign=True
        )

    def test_load_model_uses_component_keyed_state_dict_loader_for_decoder(self):
        adapter = _make_adapter()
        calibration_module = _make_module("decoder_calibration")
        embed_module = MagicMock(name="embed_calibration")
        state_dict = {"weight": MagicMock(name="weight")}
        state_dict_loader = MagicMock(return_value=state_dict)
        adapter.control_args.checkpoint = None
        adapter.model_config.repo_id = "test/repo"

        with patch.object(adapter, "_load_auto_model") as load_auto_model:
            auto_model = load_auto_model.return_value
            auto_model.get_input_embeddings.return_value.to.return_value = MagicMock(
                name="embedding_weights"
            )
            result = adapter.load_model(
                "test_model",
                extra_options={
                    "model_arch": {
                        ARTIFACT_TEXT_DECODER: {
                            DEFAULT_GRAPH_NAME: MagicMock(
                                return_value=calibration_module
                            ),
                        },
                        ARTIFACT_TOK_EMBEDDING: {
                            DEFAULT_GRAPH_NAME: MagicMock(return_value=embed_module),
                        },
                    },
                    "state_dict_loader": {ARTIFACT_TEXT_DECODER: state_dict_loader},
                },
            )

        state_dict_loader.assert_called_once_with("test/repo")
        calibration_module.load_state_dict.assert_called_once_with(
            state_dict, strict=True, assign=True
        )
        self.assertIs(
            result[ARTIFACT_TEXT_DECODER][DEFAULT_GRAPH_NAME],
            calibration_module,
        )


class TestGetExampleInputs(unittest.TestCase):
    """Multimodal graph modules own their export signatures."""

    def setUp(self):
        self.adapter = _make_adapter()

    def test_prefers_example_inputs_provided_by_the_model(self):
        expected = (MagicMock(name="tokens"), MagicMock(name="attn_mask"))
        model = MagicMock()
        model.get_example_inputs.return_value = expected

        result = self.adapter.get_example_inputs(model)

        self.assertEqual(result, expected)
        model.get_example_inputs.assert_called_once_with()

    def test_accepts_singular_model_method(self):
        expected = (MagicMock(name="input"),)
        model = MagicMock(spec=["get_example_input"])
        model.get_example_input.return_value = expected

        result = self.adapter.get_example_inputs(model)

        self.assertEqual(result, expected)
        model.get_example_input.assert_called_once_with()

    def test_requires_model_signature_instead_of_synthesizing(self):
        with self.assertRaises(ValueError):
            self.adapter.get_example_inputs(MagicMock(spec=[]))


class TestExportTokenizer(unittest.TestCase):
    """TokenizerWrapper has already written artifacts; the adapter selects the
    runtime file using the same name-based priority as the default adapter."""

    def setUp(self):
        self.adapter = _make_adapter()
        self._tmp = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self._tmp.name) / "tokenizer_artifacts"
        self.output_dir = Path(self._tmp.name) / "tokenizer_out"
        self.artifact_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _touch_artifacts(self, names):
        paths = []
        for name in names:
            path = self.artifact_dir / name
            path.touch()
            paths.append(path)
        return paths

    def _make_tokenizer(self, runtime_name):
        return MagicMock(
            runtime_tokenizer_path=str(self.artifact_dir / runtime_name),
            artifact=str(self.artifact_dir),
        )

    def test_returns_tokenizer_file_not_the_directory(self):
        self._touch_artifacts(
            [TEST_TOKENIZER_CONFIG, TEST_SPECIAL_TOKENS_MAP, TEST_TOKENIZER_JSON]
        )

        result = self.adapter.export_tokenizer(
            self._make_tokenizer(TEST_TOKENIZER_JSON), self.output_dir
        )

        self.assertEqual(result, self.artifact_dir / TEST_TOKENIZER_JSON)
        self.assertNotEqual(result, self.artifact_dir)

    def test_selects_by_name_not_by_position(self):
        self._touch_artifacts(
            [TEST_TOKENIZER_CONFIG, TEST_TOKENIZER_JSON, TEST_ADDED_TOKENS]
        )

        result = self.adapter.export_tokenizer(
            self._make_tokenizer(TEST_ADDED_TOKENS), self.output_dir
        )

        self.assertEqual(result, self.artifact_dir / TEST_TOKENIZER_JSON)

    def test_prefers_tokenizer_json_over_tokenizer_model(self):
        self._touch_artifacts([TEST_TOKENIZER_MODEL, TEST_TOKENIZER_JSON])

        result = self.adapter.export_tokenizer(
            self._make_tokenizer(TEST_TOKENIZER_MODEL), self.output_dir
        )

        self.assertEqual(result, self.artifact_dir / TEST_TOKENIZER_JSON)

    def test_falls_back_to_tokenizer_model_when_no_json(self):
        self._touch_artifacts(
            [TEST_TOKENIZER_CONFIG, TEST_TOKENIZER_MODEL, TEST_SPECIAL_TOKENS_MAP]
        )

        result = self.adapter.export_tokenizer(
            self._make_tokenizer(TEST_TOKENIZER_MODEL), self.output_dir
        )

        self.assertEqual(result, self.artifact_dir / TEST_TOKENIZER_MODEL)

    def test_falls_back_to_runtime_tokenizer_when_no_known_name(self):
        runtime_name = "tokenizer.bin"
        tokenizer = self._make_tokenizer(runtime_name)

        result = self.adapter.export_tokenizer(tokenizer, self.output_dir)

        self.assertEqual(result, self.artifact_dir / runtime_name)

    def test_appends_runtime_tokenizer_when_artifact_dir_does_not_list_it(self):
        self._touch_artifacts([TEST_TOKENIZER_CONFIG])
        runtime_path = Path(self._tmp.name) / TEST_TOKENIZER_MODEL
        runtime_path.touch()
        tokenizer = MagicMock(
            runtime_tokenizer_path=str(runtime_path),
            artifact=str(self.artifact_dir),
        )

        result = self.adapter.export_tokenizer(tokenizer, self.output_dir)

        self.assertEqual(result, runtime_path)


if __name__ == "__main__":
    unittest.main()
