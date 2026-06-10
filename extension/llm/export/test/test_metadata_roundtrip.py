# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Round-trip test: export model with metadata stored in NamedData."""

import tempfile
import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.llm.export.metadata import (
    add_metadata,
    get_float,
    get_int,
    get_int_list,
    get_string,
    read_metadata,
)
from torch.export import export


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestMetadataRoundTrip(unittest.TestCase):
    def test_roundtrip(self):
        model = SimpleModel()
        example_input = (torch.randn(1, 10),)

        exported = export(model, example_input)
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        chat_template = (
            "{% for message in messages %}"
            "{{ message.role }}: {{ message.content }}\n"
            "{% endfor %}"
        )

        add_metadata(
            edge,
            {
                "tokenizer.model": "BPE",
                "tokenizer.vocab_size": 128256,
                "chat_template": chat_template,
                "model.arch": "llama",
                "model.context_length": 8192,
                "general.name": "Llama-3.2-1B",
                "model.temperature": 0.7,
            },
        )

        et_program = edge.to_executorch()

        with tempfile.NamedTemporaryFile(suffix=".pte") as f:
            f.write(et_program.buffer)
            f.flush()

            metadata = read_metadata(f.name)

            self.assertEqual(len(metadata), 7)
            self.assertEqual(get_string(metadata, "tokenizer.model"), "BPE")
            self.assertEqual(get_int(metadata, "tokenizer.vocab_size"), 128256)
            self.assertEqual(get_string(metadata, "chat_template"), chat_template)
            self.assertEqual(get_string(metadata, "model.arch"), "llama")
            self.assertEqual(get_int(metadata, "model.context_length"), 8192)
            self.assertEqual(get_string(metadata, "general.name"), "Llama-3.2-1B")
            self.assertAlmostEqual(get_float(metadata, "model.temperature"), 0.7)

    def test_empty_metadata(self):
        model = SimpleModel()
        exported = export(model, (torch.randn(1, 10),))
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        add_metadata(edge, {})
        et_program = edge.to_executorch()

        with tempfile.NamedTemporaryFile(suffix=".pte") as f:
            f.write(et_program.buffer)
            f.flush()
            metadata = read_metadata(f.name)
            self.assertEqual(len(metadata), 0)

    def test_raw_bytes_metadata(self):
        model = SimpleModel()
        exported = export(model, (torch.randn(1, 10),))
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        raw = b"\x00\x01\x02\xff"
        add_metadata(edge, {"binary_blob": raw})
        et_program = edge.to_executorch()

        with tempfile.NamedTemporaryFile(suffix=".pte") as f:
            f.write(et_program.buffer)
            f.flush()
            metadata = read_metadata(f.name)
            self.assertEqual(metadata["binary_blob"], raw)

    def test_llm_metadata_replaces_constant_methods(self):
        """POC: these metadata fields currently live as constant_methods
        (full ExecutionPlan entries). This test shows they can be stored
        as lightweight NamedData entries instead."""
        model = SimpleModel()
        exported = export(model, (torch.randn(1, 10),))
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        add_metadata(
            edge,
            {
                "tokenizer.bos_id": 128000,
                "tokenizer.eos_ids": [128009, 128001],
                "context.max_seq_len": 8192,
                "context.max_context_len": 8192,
                "model.vocab_size": 128256,
                "model.use_kv_cache": 1,
                "model.use_sdpa_with_kv_cache": 1,
                "model.n_layers": 16,
                "tokenizer.chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
            },
        )

        et_program = edge.to_executorch()

        with tempfile.NamedTemporaryFile(suffix=".pte") as f:
            f.write(et_program.buffer)
            f.flush()

            metadata = read_metadata(f.name)

            self.assertEqual(get_int(metadata, "tokenizer.bos_id"), 128000)
            self.assertEqual(
                get_int_list(metadata, "tokenizer.eos_ids"), [128009, 128001]
            )
            self.assertEqual(get_int(metadata, "context.max_seq_len"), 8192)
            self.assertEqual(get_int(metadata, "context.max_context_len"), 8192)
            self.assertEqual(get_int(metadata, "model.vocab_size"), 128256)
            self.assertEqual(get_int(metadata, "model.use_kv_cache"), 1)
            self.assertEqual(get_int(metadata, "model.use_sdpa_with_kv_cache"), 1)
            self.assertEqual(get_int(metadata, "model.n_layers"), 16)
            self.assertIn(
                "{% for m in messages %}",
                get_string(metadata, "tokenizer.chat_template"),
            )
