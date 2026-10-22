# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K-means palettization through the full ExecuTorch lowering path.

Palettization compresses a weight into a lookup table plus an index per
element, so the observables differ from quantization: a ``lut_to_dense`` op and
``lut``/``indices`` state, rather than a scale, zero-point and quantized
weight.
"""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from executorch.backends.apple.coreai import (
    CoreAIPartitioner,
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.palettizer import CoreAIPalettizer
from executorch.exir import to_edge_transform_and_lower


def _model():
    """Deterministic, so two builds can be compared to each other."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).eval()


def _op_count(exported_program, prefix):
    """How many call_function targets start with ``prefix``.

    Matched by prefix because targets carry an overload suffix, e.g.
    ``lut_to_dense.default``.
    """
    return sum(
        1
        for node in exported_program.graph.nodes
        if node.op == "call_function"
        and getattr(node.target, "__name__", str(node.target)).startswith(prefix)
    )


def _first_lut(finalized):
    """The first palette in a finalized model, for comparing clusterings."""
    for name, tensor in finalized.state_dict().items():
        if name.endswith(".lut"):
            return tensor
    raise AssertionError("no palette in the finalized model")


@dataclass(frozen=True)
class PalettizeCase:
    """One palettization preset and the palette it should produce."""

    name: str
    preset: str
    n_bits: int

    @property
    def palette_entries(self) -> int:
        return 2**self.n_bits

    def config(self):
        from coreai_opt.palettization import KMeansPalettizerConfig

        return getattr(KMeansPalettizerConfig.presets, self.preset)()


class PalettizationTest(unittest.TestCase):
    """The presets Apple ships: w4, w6 and w8."""

    CASES = (
        PalettizeCase(name="w4", preset="w4", n_bits=4),
        PalettizeCase(name="w6", preset="w6", n_bits=6),
        PalettizeCase(name="w8", preset="w8", n_bits=8),
    )

    def setUp(self):
        self.example_inputs = (torch.randn(2, 64),)

    def _finalize(self, case):
        palettizer = CoreAIPalettizer(_model(), case.config())
        palettizer.prepare(self.example_inputs)
        return palettizer.finalize()

    def _export(self, case):
        return torch.export.export(self._finalize(case), self.example_inputs)

    def _pte_size(self, case):
        lowered = to_edge_transform_and_lower(
            self._export(case),
            transform_passes=get_default_passes(),
            partitioner=[CoreAIPartitioner()],
            compile_config=get_default_compile_config(),
        )
        return len(bytes(lowered.to_executorch().buffer))

    def test_finalize_returns_an_eager_module(self):
        """Unlike the quantizer's convert(), which returns an fx graph."""
        finalized = self._finalize(self.CASES[0])
        self.assertIsInstance(finalized, nn.Module)
        self.assertFalse(hasattr(finalized, "graph"))

    def test_weights_become_a_lookup_table(self):
        for case in self.CASES:
            with self.subTest(case.name):
                state = self._export(case).state_dict
                luts = [k for k in state if k.endswith(".lut")]
                indices = [k for k in state if k.endswith(".indices")]
                self.assertTrue(luts, f"no palette in {sorted(state)}")
                self.assertEqual(len(luts), len(indices))
                # The dense weight is emptied out in favour of the palette.
                originals = [
                    v for k, v in state.items() if k.endswith("weight.original")
                ]
                self.assertEqual(
                    len(originals),
                    len(luts),
                    f"expected one emptied weight per palette in {sorted(state)}",
                )
                self.assertTrue(all(v.numel() == 0 for v in originals))

    def test_palette_size_follows_n_bits(self):
        """A k-bit palette holds 2**k entries; guards a silently ignored preset.

        Every palette is checked, not just the first: the model has two linears,
        and a preset reaching only one of them would still be wrong.
        """
        for case in self.CASES:
            with self.subTest(case.name):
                state = self._export(case).state_dict
                luts = [v for k, v in state.items() if k.endswith(".lut")]
                self.assertTrue(luts, "no palette found")
                for lut in luts:
                    self.assertEqual(lut.shape[-2], case.palette_entries)

    def test_export_emits_lut_to_dense(self):
        for case in self.CASES:
            with self.subTest(case.name):
                self.assertGreater(_op_count(self._export(case), "lut_to_dense"), 0)

    def test_lowers_through_to_executorch(self):
        for case in self.CASES:
            with self.subTest(case.name):
                self.assertGreater(self._pte_size(case), 0)

    def test_narrower_palettes_produce_smaller_programs(self):
        sizes = {case.name: self._pte_size(case) for case in self.CASES}
        self.assertLess(sizes["w4"], sizes["w6"], sizes)
        self.assertLess(sizes["w6"], sizes["w8"], sizes)


class PalettizerApiTest(unittest.TestCase):
    def test_finalize_before_prepare_is_rejected(self):
        palettizer = CoreAIPalettizer(_model())
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            palettizer.finalize()

    def test_calibration_before_prepare_is_rejected(self):
        palettizer = CoreAIPalettizer(_model())
        with self.assertRaisesRegex(RuntimeError, "before calibration_mode"):
            palettizer.calibration_mode(loss_fn=lambda *_: torch.tensor(0.0))

    def test_finalize_twice_is_rejected(self):
        """coreai_opt clears its own prepared marker, so the guard must too."""
        palettizer = CoreAIPalettizer(_model(), PalettizationTest.CASES[0].config())
        palettizer.prepare((torch.randn(2, 64),))
        palettizer.finalize()
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            palettizer.finalize()

    def test_no_training_mode(self):
        """KMeansPalettizer does not support training-time compression.

        Forwarding to it would return a context manager that raises only on
        entry, advertising a capability that cannot work.
        """
        self.assertFalse(hasattr(CoreAIPalettizer, "training_mode"))

    def test_calibration_is_not_required(self):
        """Plain k-means clusters weights, so it needs no calibration data.

        The quantizer's calibration collects activation ranges and is
        mandatory; here it only supplies sensitivity weights.
        """
        palettizer = CoreAIPalettizer(_model(), PalettizationTest.CASES[0].config())
        palettizer.prepare((torch.randn(2, 64),))
        self.assertIsInstance(palettizer.finalize(), nn.Module)

    def test_sensitivity_calibration_changes_the_palette(self):
        """Weighted k-means must actually use the collected sensitivities."""
        sample = (torch.randn(2, 64),)
        config = PalettizationTest.CASES[0].config()

        plain = CoreAIPalettizer(_model(), config)
        plain.prepare(sample)
        plain_lut = _first_lut(plain.finalize())

        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "sensitivity")
            weighted = CoreAIPalettizer(_model(), config)
            prepared = weighted.prepare(sample)
            with weighted.calibration_mode(
                loss_fn=lambda out, target: (out - target).pow(2).mean(),
                sensitivity_path=path,
            ) as calibrate:
                calibrate.step(prepared(*sample), torch.randn(2, 64))
            weighted_lut = _first_lut(weighted.finalize())

        self.assertEqual(plain_lut.shape, weighted_lut.shape)
        self.assertFalse(
            torch.equal(plain_lut, weighted_lut),
            "sensitivity-weighted clustering produced an identical palette",
        )


if __name__ == "__main__":
    unittest.main()
