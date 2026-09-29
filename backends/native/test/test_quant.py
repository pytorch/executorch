# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.backends.native.serialization import deserialize_program
from executorch.backends.native.serialization.graph_serialize import _pack_signed_int4
from executorch.backends.native.serialization.schema import PackedQuant, ScalarType
from executorch.backends.native.test.utils import (
    _call_function_targets,
    _get_delegate_blob,
    _lower,
)
from executorch.extension.llm.export.gguf import ExportableGGUFTensor


class FuseGGUFPassTest(unittest.TestCase):
    def _packed_constants(self, program):
        return [
            c
            for c in (program.methods[0].constants or [])
            if c.meta.quant is not None and isinstance(c.meta.quant.scheme, PackedQuant)
        ]

    def test_gguf_linear_serializes_as_packed_linear(self):
        # A GGUF-quantized linear serializes as a plain linear over a weight
        # constant tagged with a PackedQuant codec, dropping the dequantize.
        n, k = 8, 256  # k must be a multiple of QK_K (256); q4_k block = 144 bytes
        blob = torch.randint(0, 256, (n, (k // 256) * 144), dtype=torch.uint8)
        lin = nn.Linear(k, n, bias=False)
        lin.weight = nn.Parameter(
            ExportableGGUFTensor.from_raw(blob, "q4_k", torch.float32),
            requires_grad=False,
        )

        program = deserialize_program(
            _get_delegate_blob(_lower(lin, (torch.randn(2, k),)))
        )
        targets = _call_function_targets(program.methods[0].graph)
        self.assertTrue(any(t and "linear" in t for t in targets))
        self.assertFalse(any(t and "dequantize_gguf" in t for t in targets))

        packed = self._packed_constants(program)
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].meta.quant.scheme.codec, "gguf:q4_k")
        self.assertEqual(packed[0].meta.dtype, ScalarType.BYTE)

    def test_activation_dtype_in_metadata(self):
        # The activation compute dtype must round-trip into the serialized
        # TensorMeta of every value in the graph (the packed weight stays BYTE).
        n, k = 8, 256  # k multiple of QK_K (256); q4_k block = 144 bytes
        blob = torch.randint(0, 256, (n, (k // 256) * 144), dtype=torch.uint8)
        for dtype, expected in (
            (torch.float32, ScalarType.FLOAT),
            (torch.float16, ScalarType.HALF),
            (torch.bfloat16, ScalarType.BFLOAT16),
        ):
            with self.subTest(dtype=dtype):
                lin = nn.Linear(k, n, bias=False)
                lin.weight = nn.Parameter(
                    ExportableGGUFTensor.from_raw(blob, "q4_k", dtype),
                    requires_grad=False,
                )
                program = deserialize_program(
                    _get_delegate_blob(_lower(lin, (torch.randn(2, k, dtype=dtype),)))
                )
                metas = [
                    tv.meta for tv in (program.methods[0].graph.tensor_values or [])
                ]
                self.assertTrue(metas, "expected serialized tensor values")
                for meta in metas:
                    self.assertEqual(meta.dtype, expected)

    def test_gguf_embedding_serializes_as_packed_embedding(self):
        # The embedding counterpart: plain embedding over a PackedQuant weight.
        num_emb, k = 16, 256  # k multiple of QK_K (256); q4_k block = 144 bytes
        blob = torch.randint(0, 256, (num_emb, (k // 256) * 144), dtype=torch.uint8)
        emb = nn.Embedding(num_emb, k)
        emb.weight = nn.Parameter(
            ExportableGGUFTensor.from_raw(blob, "q4_k", torch.float32),
            requires_grad=False,
        )

        program = deserialize_program(
            _get_delegate_blob(_lower(emb, (torch.randint(0, num_emb, (2, 3)),)))
        )
        targets = _call_function_targets(program.methods[0].graph)
        self.assertTrue(any(t and "embedding" in t for t in targets))
        self.assertFalse(any(t and "dequantize_gguf" in t for t in targets))

        packed = self._packed_constants(program)
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].meta.quant.scheme.codec, "gguf:q4_k")


class SignedInt4PackingTest(unittest.TestCase):
    def test_packs_even_value_into_low_nibble(self):
        weight = torch.tensor([[-8, -7, 6, 7]], dtype=torch.int8)

        packed = _pack_signed_int4(weight)

        torch.testing.assert_close(
            packed, torch.tensor([[0x10, 0xFE]], dtype=torch.uint8)
        )

    def test_rejects_values_outside_signed_int4_range(self):
        for value in (-9, 8):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"\[-8, 7\]"):
                    _pack_signed_int4(torch.tensor([[value, 0]], dtype=torch.int8))
