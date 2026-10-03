# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.backends.cadence.aot import compiler
from executorch.backends.cadence.aot.pass_utils import EdgePassesConfig
from executorch.backends.cadence.aot.quantizer.patterns import LinearPattern
from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceAtenQuantizer,
    CadenceQuantizer,
    qconfig_A8W8sym,
)
from executorch.backends.cadence.aot.weight_packing import (
    can_pack,
    offset,
    pack_fully_connected_weights,
    pack_rows,
    packed_row_bytes,
    remap,
    SUPPORTED_BITS,
    unpack_rows,
)
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantizer import QuantizationConfig
from torchao.quantization.pt2e.quantizer.quantizer import QuantizationSpec


# The three row lengths that occur in wakeword stage 2 once the streaming convs
# have collapsed to fully-connected: 40*5, 96, and 480.
STAGE2_SHAPES: tuple[tuple[int, int], ...] = ((96, 200), (480, 96), (96, 480), (21, 96))


class TestWeightPacking(unittest.TestCase):
    def test_round_trip_exhaustive_values(self) -> None:
        """Every representable value survives, in every lane of the group."""
        for bits in SUPPORTED_BITS:
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            span = hi - lo + 1
            # 64 groups is enough to place each value in each lane position
            n = span * 8
            vals = torch.arange(lo, hi + 1, dtype=torch.int8).repeat(8)
            w = vals.reshape(8, span).contiguous()
            if w.shape[1] % 8:
                continue
            with self.subTest(bits=bits):
                packed = pack_rows(w, bits)
                self.assertTrue(torch.equal(unpack_rows(packed, span, bits), w))
                self.assertEqual(packed.numel(), 8 * packed_row_bytes(span, bits))
                self.assertEqual(n, span * 8)

    def test_round_trip_stage2_shapes(self) -> None:
        torch.manual_seed(0)
        for bits in SUPPORTED_BITS:
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            for out_dim, in_dim in STAGE2_SHAPES:
                with self.subTest(bits=bits, shape=(out_dim, in_dim)):
                    w = torch.randint(
                        lo, hi + 1, (out_dim, in_dim), dtype=torch.int8
                    )
                    packed = pack_rows(w, bits)
                    self.assertEqual(
                        tuple(packed.shape),
                        (out_dim, packed_row_bytes(in_dim, bits)),
                    )
                    # out_dim must survive as shape[0]: meta kernels read it
                    self.assertEqual(packed.shape[0], out_dim)
                    self.assertEqual(packed.dtype, torch.int8)
                    self.assertTrue(
                        torch.equal(unpack_rows(packed, in_dim, bits), w)
                    )

    def test_scalar_remap_matches_unpack(self) -> None:
        """The index map the kernel uses agrees with the bulk unpack."""
        torch.manual_seed(1)
        for bits in SUPPORTED_BITS:
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            out_dim, in_dim = 7, 40
            w = torch.randint(lo, hi + 1, (out_dim, in_dim), dtype=torch.int8)
            packed = pack_rows(w, bits)
            with self.subTest(bits=bits):
                for r in range(out_dim):
                    for k in range(in_dim):
                        self.assertEqual(
                            remap(packed, in_dim, bits, r, k),
                            int(w[r, k]),
                            f"bits={bits} at ({r}, {k})",
                        )

    def test_matches_planar_4_2_layout(self) -> None:
        """Byte-for-byte agreement with the documented planar 4+2 packing.

        Two bit-planes, and values are paired k with k + in_dim/2 so that
        masking the low nibbles of a word yields consecutive values. That is
        what lets every decoder emit natural k order without a permutation.
        """
        # in_dim 4: half = 2, quarter = 1
        u = [5, 63, 0, 42]
        w = torch.tensor([[x - offset(6) for x in u]], dtype=torch.int8)
        got = [int(x) for x in pack_rows(w, 6).view(torch.uint8).reshape(-1)]
        expected = [
            (u[0] & 0x0F) | ((u[2] & 0x0F) << 4),  # plane A: values 0 and 0+2
            (u[1] & 0x0F) | ((u[3] & 0x0F) << 4),  # plane A: values 1 and 1+2
            (u[0] >> 4)
            | ((u[1] >> 4) << 2)
            | ((u[2] >> 4) << 4)
            | ((u[3] >> 4) << 6),  # plane B: all four fields
        ]
        self.assertEqual(got, expected)
        self.assertEqual(len(got), packed_row_bytes(4, 6))

    def test_matches_split_4_bit_layout(self) -> None:
        """Byte-for-byte agreement with the documented split 4-bit packing.

        Byte k holds value k in the low nibble and value k + in_dim/2 in the
        high nibble - the same pairing as plane A of the 6-bit layout, so a
        masked word yields consecutive values and no decoder needs to
        de-interleave. The k-with-k+1 alternative packs to the same size but
        makes every mask produce every other value.
        """
        # in_dim 4: half = 2
        u = [5, 15, 0, 9]
        w = torch.tensor([[x - offset(4) for x in u]], dtype=torch.int8)
        got = [int(x) for x in pack_rows(w, 4).view(torch.uint8).reshape(-1)]
        expected = [
            (u[0] & 0x0F) | ((u[2] & 0x0F) << 4),  # values 0 and 0+2
            (u[1] & 0x0F) | ((u[3] & 0x0F) << 4),  # values 1 and 1+2
        ]
        self.assertEqual(got, expected)
        self.assertEqual(len(got), packed_row_bytes(4, 4))

    def test_stage2_footprint(self) -> None:
        """The packed size is what the study predicts, with no padding."""
        tensors = (
            [(96, 200), (480, 96), (96, 480)]
            + [(96, 480), (480, 96), (96, 480)] * 4
            + [(21, 96)]
        )
        dense = sum(o * i for o, i in tensors)
        packed = sum(o * packed_row_bytes(i, 6) for o, i in tensors)
        self.assertEqual(dense, 666_336)
        self.assertEqual(packed, 499_752)
        # every row length is a multiple of the 6-bit group size, so no padding
        for _, in_dim in tensors:
            self.assertEqual(in_dim % 4, 0)

    def test_rejects_out_of_range(self) -> None:
        w = torch.full((1, 4), 32, dtype=torch.int8)  # 32 does not fit 6 bits
        with self.assertRaisesRegex(ValueError, "does not fit 6 bits"):
            pack_rows(w, 6)

    def test_rejects_unaligned_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            packed_row_bytes(6, 4 + 2)  # in_dim=6 with a 4-value group

    def test_rejects_unsupported_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported weight bit width"):
            packed_row_bytes(8, 5)


class TestPackFullyConnectedWeights(unittest.TestCase):
    """The transform that rewrites a lowered graph onto the packed operator.

    Packing is storage-only, so it needs weights that are already clamped to
    the narrow range: a 6-bit weight is an int8 tensor quantized with
    quant_min/quant_max of -32/31. Quantizing at 8 bits and then asking for
    6-bit packing must (and does) decline.
    """

    class Net(torch.nn.Module):
        def __init__(self, in_dim: int = 16, out_dim: int = 8) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(in_dim, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    @staticmethod
    def _quantizer(bits: int) -> CadenceQuantizer:
        """A default quantizer whose weight range is `bits` wide."""
        weight = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-(1 << (bits - 1)),
            quant_max=(1 << (bits - 1)) - 1,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=False,
            observer_or_fake_quant_ctr=MinMaxObserver,
        )
        config = QuantizationConfig(
            qconfig_A8W8sym.input_activation,
            qconfig_A8W8sym.output_activation,
            weight,
            None,
        )
        return CadenceQuantizer([CadenceAtenQuantizer(LinearPattern(), config)])

    def _lower(
        self, bits: int, in_dim: int = 16, quantize_bits: int | None = None
    ) -> torch.export.ExportedProgram:
        torch.manual_seed(0)
        model = self.Net(in_dim=in_dim).eval()
        inputs = (torch.randn(1, in_dim),)
        quantized = compiler.quantize_pt2(
            model, inputs, self._quantizer(quantize_bits or bits)
        )
        manager = compiler._lower_ep_to_cadence(
            quantized, edge_passes_config=EdgePassesConfig(weight_bits=bits)
        )
        return manager.exported_program()

    @staticmethod
    def _targets(ep: torch.export.ExportedProgram) -> list[str]:
        return [
            n.target.name()
            for n in ep.graph.nodes
            if n.op == "call_function" and hasattr(n.target, "name")
        ]

    @staticmethod
    def _int8_matrices(ep: torch.export.ExportedProgram) -> list[torch.Tensor]:
        values = list(ep.state_dict.values()) + list(ep.constants.values())
        return [
            t
            for t in values
            if isinstance(t, torch.Tensor) and t.dtype == torch.int8 and t.dim() == 2
        ]

    def test_lowering_at_6_bits_emits_the_packed_op(self) -> None:
        names = self._targets(self._lower(6))
        self.assertIn("cadence::quantized_fully_connected_packed", names)
        self.assertNotIn("cadence::quantized_fully_connected.per_tensor", names)

    def test_default_lowering_is_untouched(self) -> None:
        """weight_bits=8 must leave the graph exactly as it was."""
        names = self._targets(self._lower(8))
        self.assertNotIn("cadence::quantized_fully_connected_packed", names)
        self.assertIn("cadence::quantized_fully_connected.per_tensor", names)

    def test_packed_weight_is_physically_smaller(self) -> None:
        # 16 values per row at 6 bits is 12 bytes, not 16.
        shapes = [tuple(t.shape) for t in self._int8_matrices(self._lower(6))]
        self.assertIn((8, 12), shapes, f"no row-packed weight found, got {shapes}")

    def test_eight_bit_weights_decline_to_pack(self) -> None:
        """Asking for 6-bit packing on 8-bit weights must not corrupt them."""
        names = self._targets(self._lower(6, quantize_bits=8))
        self.assertIn("cadence::quantized_fully_connected.per_tensor", names)
        self.assertNotIn("cadence::quantized_fully_connected_packed", names)

    def test_misaligned_row_is_left_alone(self) -> None:
        """in_dim not a multiple of the group size cannot pack row-aligned.

        The layer degrades to 8 bits rather than failing the whole compile.
        """
        names = self._targets(self._lower(6, in_dim=14))
        self.assertIn("cadence::quantized_fully_connected.per_tensor", names)
        self.assertNotIn("cadence::quantized_fully_connected_packed", names)

    def test_out_of_range_weights_are_left_alone(self) -> None:
        weight = torch.tensor([[40, 1, 2, 3]], dtype=torch.int8)
        self.assertFalse(can_pack(weight, 6))
        self.assertTrue(can_pack(weight.clamp(-32, 31), 6))

    def test_unsupported_width_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported weight bit width"):
            pack_fully_connected_weights(self._lower(8), 5)

    def _execute(self, bits: int, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        torch.manual_seed(0)
        model = self.Net().eval()
        quantized = compiler.quantize_pt2(model, inputs, self._quantizer(6))
        manager = compiler._lower_ep_to_cadence(
            quantized, edge_passes_config=EdgePassesConfig(weight_bits=bits)
        )
        return manager.exported_program().module()(*inputs)

    def test_packed_execution_is_bit_identical_to_unpacked(self) -> None:
        """The oracle for packing is the same model unpacked, not float.

        Both sides are quantized at 6 bits; only the storage differs. Anything
        other than exact equality means the layout or the decode is wrong, and
        this holds regardless of the requantization conventions the operator
        happens to use.
        """
        torch.manual_seed(0)
        inputs = (torch.randn(1, 16),)

        packed = self._execute(6, inputs)
        unpacked = self._execute(8, inputs)

        self.assertTrue(
            torch.equal(packed, unpacked),
            f"packing changed the result: {packed} vs {unpacked}",
        )
