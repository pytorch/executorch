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
from executorch.backends.native.serialization.schema import (
    AffineGroup,
    PackedQuant,
    ScalarType,
    TensorArg,
)
from executorch.backends.native.test.utils import (
    _call_function_targets,
    _get_delegate_blob,
    _lower,
)
from executorch.exir.native import to_native
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup


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


def _weight_of(method, op="aten.linear"):
    [node] = [n for n in method.graph.nodes if n.target and op in n.target]
    [weight] = [a.arg.value for a in node.inputs if a.name == "weight"]
    assert isinstance(weight, TensorArg)
    return weight.name


class TorchaoQ4LinearTest(unittest.TestCase):
    N, K, GROUP_SIZE = 16, 64, 32

    def _serialize(self, config):
        model = nn.Sequential(nn.Linear(self.K, self.N, bias=False))
        quantize_(model, config)
        manager = to_native(torch.export.export(model, (torch.randn(2, self.K),)))
        return deserialize_program(manager._ptg).methods[0], manager._constants

    def _assert_linear_reads_packed_weight(self, method, constants):
        [weight] = [c for c in method.constants if c.meta.quant is not None]
        scheme = weight.meta.quant.scheme
        self.assertIsInstance(scheme, AffineGroup)
        self.assertEqual(_weight_of(method), weight.name)
        self.assertEqual(weight.meta.dtype, ScalarType.BYTE)
        self.assertEqual([d.max for d in weight.meta.sizes], [self.N, self.K])
        self.assertEqual(
            (scheme.quant_min, scheme.quant_max, scheme.group_size),
            (-8, 7, self.GROUP_SIZE),
        )
        self.assertEqual(scheme.scale_dtype, ScalarType.FLOAT)
        self.assertEqual(constants[weight.data_key].dtype, torch.uint8)
        self.assertEqual(constants[weight.data_key].numel(), self.N * self.K // 2)
        self.assertIn(scheme.scale_data_key, constants)
        self.assertIn(scheme.zero_point_data_key, constants)

    def test_8da4w_linear_reads_packed_weight(self):
        method, constants = self._serialize(
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4, weight_granularity=PerGroup(self.GROUP_SIZE)
            )
        )
        targets = _call_function_targets(method.graph)
        # Only the activation keeps its choose_qparams -> quantize -> dequantize.
        self.assertEqual(sum("dequantize_affine" in t for t in targets), 1)
        self.assertEqual(sum("quantize_affine" in t for t in targets), 2)
        self.assertTrue(any("choose_qparams_affine" in t for t in targets))
        self._assert_linear_reads_packed_weight(method, constants)

    def test_4w_linear_reads_packed_weight(self):
        method, constants = self._serialize(
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4, granularity=PerGroup(self.GROUP_SIZE)
            )
        )
        targets = _call_function_targets(method.graph)
        self.assertFalse(any("quantize_affine" in t for t in targets))
        self._assert_linear_reads_packed_weight(method, constants)


class TorchaoQ4EmbeddingTest(unittest.TestCase):
    def test_embedding_reads_packed_weight(self):
        rows, cols, group_size = 16, 64, 32
        model = nn.Sequential(nn.Embedding(rows, cols))
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4, granularity=PerGroup(group_size)
            ),
            filter_fn=lambda m, _: isinstance(m, nn.Embedding),
        )
        indices = torch.tensor([[1, 5, 7]])
        manager = to_native(torch.export.export(model, (indices,)))
        method = deserialize_program(manager._ptg).methods[0]

        self.assertFalse(
            any("dequantize_affine" in t for t in _call_function_targets(method.graph))
        )
        [weight] = [c for c in method.constants if c.meta.quant is not None]
        scheme = weight.meta.quant.scheme
        self.assertEqual(_weight_of(method, "aten.embedding"), weight.name)
        self.assertEqual([d.max for d in weight.meta.sizes], [rows, cols])
        self.assertEqual(
            (scheme.quant_min, scheme.quant_max, scheme.group_size),
            (-8, 7, group_size),
        )
        self.assertEqual(manager._constants[weight.data_key].numel(), rows * cols // 2)


class _DequantizedLinear(nn.Module):
    """linear(x, dequantize_affine(weight)) over an int4 [8, 16] weight in groups
    of 8, optionally with the weight as an input or the dequantize read twice."""

    def __init__(self, output_dtype=torch.float32, weight_input=False, reuse=False):
        super().__init__()
        self.output_dtype = output_dtype
        self.weight_input = weight_input
        self.reuse = reuse
        self.register_buffer("weight", torch.randint(-8, 8, (8, 16), dtype=torch.int8))
        self.register_buffer("scale", torch.rand(8, 2) + 0.5)
        self.register_buffer("zero_point", torch.zeros(8, 2, dtype=torch.int8))

    def example_inputs(self):
        x = torch.randn(2, 16, dtype=self.output_dtype)
        return (x, self.weight.clone()) if self.weight_input else (x,)

    def forward(self, x, weight=None):
        weight = self.weight if weight is None else weight
        dequantized = torch.ops.torchao.dequantize_affine(
            weight,
            [1, 8],
            self.scale,
            self.zero_point,
            torch.int8,
            -8,
            7,
            output_dtype=self.output_dtype,
        )
        out = torch.nn.functional.linear(x, dequantized)
        return out + dequantized.sum() if self.reuse else out


class FoldTorchaoQ4DequantizeTest(unittest.TestCase):
    def _method(self, model):
        return deserialize_program(
            _get_delegate_blob(_lower(model, model.example_inputs()))
        ).methods[0]

    def _weight_dequantizes(self, model):
        targets = _call_function_targets(self._method(model).graph)
        return sum("dequantize_affine" in t for t in targets)

    def _packed_constants(self, model):
        return [c for c in self._method(model).constants if c.meta.quant is not None]

    def test_folds_constant_weight_read_by_linear(self):
        self.assertEqual(self._weight_dequantizes(_DequantizedLinear()), 0)
        self.assertEqual(len(self._packed_constants(_DequantizedLinear())), 1)

    def test_keeps_unfolded_weight_unpacked(self):
        for model in (
            _DequantizedLinear(reuse=True),
            _DequantizedLinear(output_dtype=torch.float16),
        ):
            self.assertEqual(self._packed_constants(model), [])

    def test_keeps_dequantize_of_runtime_weight(self):
        self.assertEqual(
            self._weight_dequantizes(_DequantizedLinear(weight_input=True)), 1
        )

    def test_keeps_dequantize_read_by_another_op(self):
        self.assertEqual(self._weight_dequantizes(_DequantizedLinear(reuse=True)), 1)

    def test_keeps_dequantize_to_another_dtype_than_scales(self):
        self.assertEqual(
            self._weight_dequantizes(_DequantizedLinear(output_dtype=torch.float16)),
            1,
        )
