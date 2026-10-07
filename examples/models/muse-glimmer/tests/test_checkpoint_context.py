# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint metadata and mocked export routing; no model construction."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

from executorch.examples.models.muse_glimmer.export import export_solo
from executorch.examples.models.muse_glimmer.loaders import checkpoint_loader as loaders
from executorch.examples.models.muse_glimmer.model import model as model_module
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig
from executorch.extension.llm.export import load, quant


def _require_gguf(testcase):
    try:
        import gguf
    except ImportError as e:
        testcase.skipTest(f"gguf package required: {e}")
    return gguf


class CheckpointContextTest(unittest.TestCase):
    def setUp(self):
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        self.root = Path(self.contexts.enter_context(tempfile.TemporaryDirectory()))
        self.finalize = self.contexts.enter_context(
            patch.object(loaders, "_finalize", return_value=sentinel.model)
        )

    def write_json(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return str(path)

    def write_gguf(self, fields, architecture="muse-glimmer"):
        path = self.root / "metadata.gguf"
        writer = _require_gguf(self).GGUFWriter(path, architecture)
        for key, value, value_type in fields:
            writer.add_key_value(key, value, value_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        return str(path)

    def test_gguf_architecture_context(self):
        GGUFValueType = _require_gguf(self).GGUFValueType
        path = self.write_gguf(
            [
                ("test-architecture.context_length", 131072, GGUFValueType.UINT32),
                ("llama.context_length", 32, GGUFValueType.UINT32),
            ],
            architecture="test-architecture",
        )
        with patch.object(loaders, "_atomic_sd_from_gguf", return_value={}):
            _, config = loaders.load_gguf_model(path, max_seq_len=None)
        self.assertEqual(config.max_seq_len, 131072)

    def test_mlx_nested_context_and_fallback(self):
        for text_config, expected in (
            ({"max_position_embeddings": 131072}, 131072),
            ({}, 4096),
        ):
            with self.subTest(text_config=text_config):
                self.write_json(
                    "config.json",
                    {"max_position_embeddings": 4096, "text_config": text_config},
                )
                with patch.object(load, "iter_checkpoint", return_value=[]):
                    _, config = loaders.load_mlx_model(str(self.root), max_seq_len=None)
                self.assertEqual(config.max_seq_len, expected)

    def test_params_loaders_use_checkpoint_context(self):
        self.write_json("params.json", {"max_seq_len": 4096})
        self.contexts.enter_context(
            patch.object(loaders, "_atomic_sd_from_safetensors", return_value={})
        )
        self.contexts.enter_context(
            patch.object(model_module, "_load_and_remap_checkpoint", return_value={})
        )
        self.contexts.enter_context(
            patch.object(quant, "quantize_stream", return_value=[])
        )
        for loader, args in (
            (loaders.load_prequantized_model, (str(self.root),)),
            (loaders.load_and_quantize, (str(self.root), sentinel.recipe)),
        ):
            with self.subTest(loader=loader.__name__):
                model, config = loader(
                    *args, max_seq_len=None, defer_runtime_buffers=True
                )
                self.assertIs(model, sentinel.model)
                self.assertEqual(config.max_seq_len, 4096)
                self.assertTrue(self.finalize.call_args.kwargs["defer_runtime_buffers"])

    def test_invalid_gguf_context_fails_before_weights(self):
        with patch.object(loaders, "_atomic_sd_from_gguf") as weights:
            with self.assertRaisesRegex(ValueError, "Native context"):
                loaders.load_gguf_model(self.write_gguf([]), max_seq_len=None)
            weights.assert_not_called()
        self.finalize.assert_not_called()

    def test_invalid_context_fails_before_weights(self):
        self.write_json(
            "config.json",
            {
                "max_position_embeddings": 4096,
                "text_config": {"max_position_embeddings": True},
            },
        )
        with patch.object(load, "iter_checkpoint") as weights:
            with self.assertRaisesRegex(ValueError, "Native context"):
                loaders.load_mlx_model(str(self.root), max_seq_len=None)
            weights.assert_not_called()
        self.write_json("params.json", {"max_seq_len": -1})
        with patch.object(model_module, "_load_and_remap_checkpoint") as weights:
            with self.assertRaisesRegex(ValueError, "Native context"):
                loaders.load_and_quantize(
                    str(self.root), sentinel.recipe, max_seq_len=None
                )
            weights.assert_not_called()
        self.finalize.assert_not_called()


class BatchingExportContextTest(unittest.TestCase):
    SOURCES = (
        ("--gguf", "load_gguf_model"),
        ("--mlx", "load_mlx_model"),
        ("--prequantized", "load_prequantized_model"),
        ("--checkpoint-dir", "load_and_quantize"),
    )

    def setUp(self):
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        self.stderr = self.contexts.enter_context(
            contextlib.redirect_stderr(io.StringIO())
        )
        self.config = MuseGlimmerConfig(max_seq_len=4096)
        self.loaders = {
            name: self.contexts.enter_context(
                patch.object(loaders, name, return_value=(sentinel.model, self.config))
            )
            for _, name in self.SOURCES
        }
        self.export = self.contexts.enter_context(
            patch.object(export_solo, "export_and_lower")
        )

    def run_cli(self, source, *, offgraph=True, prefill=128):
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
        ]
        if offgraph:
            argv.append("--use-offgraph-kv-cache")
        with patch.object(sys, "argv", argv):
            export_solo.main()

    def test_offgraph_context_and_prefill_routing(self):
        for source, name in self.SOURCES:
            with self.subTest(source=source):
                self.export.reset_mock()
                self.run_cli(source)
                bound = self.loaders[name].call_args
                # The params-loader wrappers pass max_seq_len positionally.
                limit = (
                    bound.kwargs["max_seq_len"]
                    if source in ("--gguf", "--mlx")
                    else bound.args[1 if source == "--prequantized" else 2]
                )
                self.assertIsNone(limit)
                self.assertTrue(bound.kwargs["defer_runtime_buffers"])
                self.assertIs(self.export.call_args.args[1], self.config)
                self.assertEqual(self.config.max_seq_len, 4096)
                self.assertEqual(self.export.call_args.kwargs["max_prefill_chunk"], 128)
                self.assertTrue(self.export.call_args.kwargs["use_offgraph_kv_cache"])

    def test_invalid_prefill_stops_export(self):
        for prefill, message in (
            (0, "must be positive"),
            (4097, "native context limit (4096)"),
        ):
            with self.subTest(prefill=prefill):
                with self.assertRaises(SystemExit) as error:
                    self.run_cli("--gguf", prefill=prefill)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, self.stderr.getvalue())
                if prefill == 0:
                    self.loaders["load_gguf_model"].assert_not_called()
        self.export.assert_not_called()

    def test_legacy_explicit_context(self):
        self.run_cli("--gguf", offgraph=False)
        bound = self.loaders["load_gguf_model"].call_args
        self.assertEqual(bound.kwargs["max_seq_len"], 32)
        self.assertFalse(bound.kwargs.get("defer_runtime_buffers", False))
        self.assertFalse(self.export.call_args.kwargs["use_offgraph_kv_cache"])


class CUDAExportContextTest(unittest.TestCase):
    def test_offgraph_cli_dispatch(self):
        config = MuseGlimmerConfig(max_seq_len=32)
        argv = "export_solo --gguf unused --backend cuda --max-seq-len 32 --use-offgraph-kv-cache".split()
        with (
            patch.object(sys, "argv", argv),
            patch.object(export_solo.torch.cuda, "is_available", return_value=True),
            patch.object(
                loaders, "load_gguf_model", return_value=(sentinel.model, config)
            ) as loader,
            patch.object(export_solo, "_export_cuda") as export,
        ):
            export_solo.main()
        self.assertEqual(loader.call_args.kwargs["max_seq_len"], 32)
        self.assertNotIn("defer_runtime_buffers", loader.call_args.kwargs)
        export.assert_called_once()
        self.assertEqual(export.call_args.args[:2], (sentinel.model, config))
        self.assertTrue(export.call_args.kwargs["use_offgraph_kv_cache"])
