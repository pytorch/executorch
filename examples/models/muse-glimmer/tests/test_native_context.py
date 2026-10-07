# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metadata-only and mocked-dispatch tests; no model/export/runtime execution."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

import torch
from executorch.examples.models.muse_glimmer.export import export_solo
from executorch.examples.models.muse_glimmer.loaders import (
    checkpoint_loader as loaders,
    quantize_and_save,
)
from executorch.examples.models.muse_glimmer.model import model as model_module
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig
from executorch.extension.llm.export import load, quant
from gguf import GGUFReader, GGUFValueType, GGUFWriter


class NativeContextTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(
            patch.object(
                model_module, "MuseGlimmerModel", side_effect=AssertionError("model")
            )
        )
        self.finalize = self.enterContext(
            patch.object(loaders, "_finalize", return_value=sentinel.model)
        )
        self.gguf_weights = self.enterContext(
            patch.object(loaders, "_atomic_sd_from_gguf", return_value={})
        )
        self.safetensors = self.enterContext(
            patch.object(loaders, "_atomic_sd_from_safetensors", return_value={})
        )
        self.iter_checkpoint = self.enterContext(
            patch.object(load, "iter_checkpoint", return_value=[])
        )
        self.bf16_weights = self.enterContext(
            patch.object(model_module, "_load_and_remap_checkpoint", return_value={})
        )
        self.enterContext(
            patch.object(model_module, "_split_fused_qkv", return_value={})
        )
        self.enterContext(patch.object(model_module, "_split_gate_up", return_value={}))
        self.quantize = self.enterContext(
            patch.object(quant, "quantize_stream", return_value=[])
        )

    def write_json(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return str(path)

    def write_gguf(self, fields, architecture="muse-glimmer"):
        path = self.root / "metadata.gguf"
        writer = GGUFWriter(path, architecture)
        for key, value, value_type in fields:
            writer.add_key_value(key, value, value_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        self.assertEqual(len(GGUFReader(path).tensors), 0)
        return str(path)

    def test_gguf_uses_declared_architecture_context(self):
        for architecture in ("muse-glimmer", "test-architecture"):
            with self.subTest(architecture=architecture):
                path = self.write_gguf(
                    [
                        (
                            f"{architecture}.context_length",
                            131072,
                            GGUFValueType.UINT32,
                        ),
                        ("llama.context_length", 32, GGUFValueType.UINT32),
                    ],
                    architecture,
                )
                model, config = loaders.load_gguf_model(
                    path, backend="mlx", max_seq_len=None, defer_runtime_buffers=True
                )
                self.assertIs(model, sentinel.model)
                self.assertEqual(config.max_seq_len, 131072)
                self.assertIs(self.finalize.call_args.args[2], config)
                self.assertTrue(self.finalize.call_args.kwargs["defer_runtime_buffers"])

    def test_gguf_rejects_missing_or_invalid_context(self):
        cases = [
            [],
            [("llama.context_length", 4096, GGUFValueType.UINT32)],
            [("muse-glimmer.context_length", 0, GGUFValueType.INT32)],
            [("muse-glimmer.context_length", -1, GGUFValueType.INT32)],
            [("muse-glimmer.context_length", True, GGUFValueType.BOOL)],
            [("muse-glimmer.context_length", "4096", GGUFValueType.STRING)],
            [("muse-glimmer.context_length", 4096.0, GGUFValueType.FLOAT32)],
            [("muse-glimmer.context_length", [4096], GGUFValueType.ARRAY)],
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                path = self.write_gguf(fields)
                with self.assertRaisesRegex(
                    ValueError, "Native context.*context_length"
                ):
                    loaders.load_gguf_model(path, max_seq_len=None)
        self.gguf_weights.assert_not_called()
        self.finalize.assert_not_called()

    def test_gguf_rejects_missing_or_invalid_architecture(self):
        # An empty architecture asks GGUFWriter not to emit general.architecture.
        for fields in (
            [],
            [("general.architecture", " ", GGUFValueType.STRING)],
            [("general.architecture", 1, GGUFValueType.UINT32)],
        ):
            with self.subTest(fields=fields):
                path = self.write_gguf(fields, architecture="")
                with self.assertRaisesRegex(ValueError, "general.architecture"):
                    loaders.load_gguf_model(path, max_seq_len=None)
        self.gguf_weights.assert_not_called()

    def test_mlx_nested_and_top_level_context(self):
        for data, expected in (
            ({"max_position_embeddings": 4096}, 4096),
            ({"text_config": {"max_position_embeddings": 131072}}, 131072),
            (
                {
                    "max_position_embeddings": 32,
                    "text_config": {"max_position_embeddings": 4096},
                },
                4096,
            ),
            ({"max_position_embeddings": 4096, "text_config": {}}, 4096),
        ):
            with self.subTest(data=data):
                self.write_json("config.json", data)
                _, config = loaders.load_mlx_model(
                    str(self.root), max_seq_len=None, defer_runtime_buffers=True
                )
                self.assertEqual(config.max_seq_len, expected)
                self.assertIs(self.finalize.call_args.args[2], config)
                self.assertTrue(self.finalize.call_args.kwargs["defer_runtime_buffers"])

    def test_mlx_invalid_context_does_not_fall_back(self):
        cases = [{}, [], {"text_config": None}, {"text_config": []}]
        for value in (None, 0, -1, True, False, "4096", 4096.0, [4096]):
            cases.append({"max_position_embeddings": value})
            cases.append(
                {
                    "max_position_embeddings": 4096,
                    "text_config": {"max_position_embeddings": value},
                }
            )
        for data in cases:
            with self.subTest(data=data):
                self.write_json("config.json", data)
                with self.assertRaisesRegex(ValueError, "Native context"):
                    loaders.load_mlx_model(str(self.root), max_seq_len=None)
        self.iter_checkpoint.assert_not_called()
        self.finalize.assert_not_called()

    def test_required_params_context_and_legacy_reads(self):
        path = self.write_json("params.json", {"max_seq_len": 4096, "extra": 1})
        with patch.object(model_module.json, "load", wraps=json.load) as read:
            config = MuseGlimmerConfig.from_json(path, require_native_context=True)
        self.assertEqual(config.max_seq_len, 4096)
        read.assert_called_once()
        path = self.write_json("params.json", {})
        self.assertEqual(MuseGlimmerConfig.from_json(path).max_seq_len, 16384)
        for data in (
            {},
            [],
            *(
                {"max_seq_len": v}
                for v in (None, 0, -1, True, False, "4096", 4096.0, [4096])
            ),
        ):
            with self.subTest(data=data):
                path = self.write_json("params.json", data)
                with self.assertRaisesRegex(ValueError, "Native context.*max_seq_len"):
                    MuseGlimmerConfig.from_json(path, require_native_context=True)
        path = self.write_json("params.json", {"max_seq_len": "legacy"})
        self.assertEqual(MuseGlimmerConfig.from_json(path).max_seq_len, "legacy")

    def test_params_loaders_preserve_native_context_and_numeric_defaults(self):
        self.write_json("params.json", {"max_seq_len": 4096})
        for loader, args in (
            (loaders.load_prequantized_model, (str(self.root),)),
            (loaders.load_and_quantize, (str(self.root), sentinel.recipe)),
        ):
            for kwargs, expected in (
                ({}, 16384),
                ({"max_seq_len": 32}, 32),
                ({"max_seq_len": None}, 4096),
            ):
                with self.subTest(loader=loader.__name__, kwargs=kwargs):
                    model, config = loader(*args, **kwargs, defer_runtime_buffers=True)
                    self.assertIs(model, sentinel.model)
                    self.assertEqual(config.max_seq_len, expected)
                    self.assertIs(self.finalize.call_args.args[2], config)
                    self.assertTrue(
                        self.finalize.call_args.kwargs["defer_runtime_buffers"]
                    )
        self.quantize.assert_called()

    def test_params_loaders_fail_before_weights_when_native_context_missing(self):
        self.write_json("params.json", {})
        for loader, args in (
            (loaders.load_prequantized_model, (str(self.root),)),
            (loaders.load_and_quantize, (str(self.root), sentinel.recipe)),
        ):
            with self.subTest(loader=loader.__name__):
                with self.assertRaisesRegex(ValueError, "Native context.*max_seq_len"):
                    loader(*args, max_seq_len=None)
        self.safetensors.assert_not_called()
        self.bf16_weights.assert_not_called()
        self.quantize.assert_not_called()
        self.finalize.assert_not_called()

    def test_gguf_mlx_defaults_overrides_and_explicit_config_priority(self):
        # Missing metadata is deliberately never opened for numeric or explicit config loads.
        for loader in (loaders.load_gguf_model, loaders.load_mlx_model):
            with self.subTest(loader=loader.__name__):
                _, config = loader(str(self.root))
                self.assertEqual(config.max_seq_len, 131072)
                _, config = loader(str(self.root), max_seq_len=32)
                self.assertEqual(config.max_seq_len, 32)
                explicit = MuseGlimmerConfig(max_seq_len=777)
                for override in (None, 32):
                    _, config = loader(
                        str(self.root), max_seq_len=override, config=explicit
                    )
                    self.assertIs(config, explicit)
                    self.assertEqual(config.max_seq_len, 777)
                self.assertFalse(
                    self.finalize.call_args.kwargs["defer_runtime_buffers"]
                )


class NativeContextCLITest(unittest.TestCase):
    SOURCES = (
        ("--gguf", "load_gguf_model"),
        ("--mlx", "load_mlx_model"),
        ("--prequantized", "load_prequantized_model"),
        ("--checkpoint-dir", "load_and_quantize"),
    )

    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.config = MuseGlimmerConfig(max_seq_len=4096)
        self.loaders = {
            name: self.enterContext(
                patch.object(loaders, name, return_value=(sentinel.model, self.config))
            )
            for _, name in self.SOURCES
        }
        self.enterContext(
            patch.object(
                quantize_and_save,
                "build_recipes",
                return_value={"default": sentinel.recipe},
            )
        )
        self.vision = self.enterContext(
            patch.object(
                loaders,
                "load_mmproj_vision_model",
                return_value=(sentinel.vision, sentinel.positions, None),
            )
        )
        self.export = self.enterContext(patch.object(export_solo, "export_and_lower"))
        self.enterContext(
            patch.object(
                model_module, "MuseGlimmerModel", side_effect=AssertionError("model")
            )
        )

    def run_cli(self, source, *, native=True, prefill=128):
        argv = [
            "export_solo",
            source,
            "unused-checkpoint",
            "--backend",
            "mlx",
            "--max-seq-len",
            "32",
            "--max-prefill-chunk",
            str(prefill),
            "--mmproj",
            "unused-vision",
        ]
        if native:
            argv.append("--use-offgraph-kv-cache")
        with patch.object(sys, "argv", argv):
            export_solo.main()

    def test_all_sources_use_native_context_and_keep_prefill_separate(self):
        for source, name in self.SOURCES:
            with self.subTest(source=source):
                self.run_cli(source)
                loader = self.loaders[name]
                bound = loader.call_args
                # The two wrapper routes pass max_seq_len positionally.
                limit = (
                    bound.kwargs["max_seq_len"]
                    if source in ("--gguf", "--mlx")
                    else bound.args[1 if source == "--prequantized" else 2]
                )
                self.assertIsNone(limit)
                self.assertTrue(bound.kwargs["defer_runtime_buffers"])
                self.assertIs(self.export.call_args.args[1], self.config)
                self.assertEqual(self.export.call_args.kwargs["max_prefill_chunk"], 128)
                self.assertTrue(self.export.call_args.kwargs["use_offgraph_kv_cache"])
                loader.assert_called_once()
                self.vision.assert_called_once()
                self.export.assert_called_once()
                loader.reset_mock()
                self.vision.reset_mock()
                self.export.reset_mock()

    def test_all_sources_keep_legacy_numeric_override(self):
        for source, name in self.SOURCES:
            with self.subTest(source=source):
                self.run_cli(source, native=False)
                bound = self.loaders[name].call_args
                limit = (
                    bound.kwargs["max_seq_len"]
                    if source in ("--gguf", "--mlx")
                    else bound.args[1 if source == "--prequantized" else 2]
                )
                self.assertEqual(limit, 32)
                self.assertFalse(bound.kwargs.get("defer_runtime_buffers", False))
                self.assertFalse(self.export.call_args.kwargs["use_offgraph_kv_cache"])

    def test_prefill_overflow_fails_before_vision_or_export(self):
        for source, name in self.SOURCES:
            with self.subTest(source=source):
                with self.assertRaises(SystemExit) as error:
                    self.run_cli(source, prefill=4097)
                self.assertEqual(error.exception.code, 2)
                self.loaders[name].assert_called_once()
        self.assertIn("native context limit (4096)", self.stderr.getvalue())
        self.vision.assert_not_called()
        self.export.assert_not_called()

    def test_metadata_separates_native_context_from_forward_width(self):
        metadata = export_solo._solo_constant_methods(
            config=self.config,
            max_prefill=128,
            activation_dtype=torch.float16,
            mutable_buffer_metadata=None,
            has_vision=False,
            max_vision_patches=16384,
            use_offgraph_kv_cache=True,
        )
        self.assertEqual(metadata["get_max_context_len"], 4096)
        self.assertEqual(metadata["get_max_seq_len"], 128)
        self.assertEqual(metadata["get_max_prefill_chunk"], 128)

    def test_nonpositive_prefill_fails_before_loading(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--gguf", prefill=0)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("must be positive", self.stderr.getvalue())
        for loader in self.loaders.values():
            loader.assert_not_called()
        self.vision.assert_not_called()
        self.export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
