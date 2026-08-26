# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Edge behaviour of the activation lookup table.

The end-to-end tests compare against a quantized reference, which clamps a pole
to the same rail whatever the table holds, so they agree there by construction.
These pin the entries directly, which is where the domain handling lives.
"""

import unittest

import torch
from executorch.backends.cortex_m.passes.aten_to_cortex_m_pass import (
    _get_activation_replacement,
    AtenToCortexMPass,
)
from executorch.backends.cortex_m.passes.passes_utils import (
    _ACTIVATION_FNS,
    _round_half_away_from_zero,
    build_activation_lut,
)
from executorch.backends.cortex_m.quantizer.quantizer_support import (
    ACTIVATION_OP_PATTERNS,
)
from executorch.exir.dialects._ops import ops as exir_ops

aten = exir_ops.edge.aten


def _functional_edge_target(aten_op):
    """The edge op an aten activation lowers to. The in-place spelling is
    functionalized onto the same one, so both map here."""
    name = aten_op._schema.name.split("::")[1].rstrip("_")
    return getattr(aten, name).default


class ActivationLutEdgeTests(unittest.TestCase):
    def _entry(self, target, x, input_scale, input_zp, output_scale, output_zp):
        """The table entry an input of `x` would index."""
        q = round(x / input_scale) + input_zp
        self.assertTrue(-128 <= q <= 127, f"{x} is outside the int8 range")
        lut = build_activation_lut(
            target, input_scale, input_zp, output_scale, output_zp
        )
        return int(lut[q + 128])

    def test_log_family_saturates_at_the_pole(self):
        # Zero is always exactly representable in an affine int8 quantizer, so
        # this entry exists in every log table the backend builds.
        for target in (aten.log.default, aten.log2.default, aten.log10.default):
            with self.subTest(target=str(target)):
                self.assertEqual(
                    self._entry(target, 0.0, 0.01568, -128, 0.01624, 42), -128
                )

    def test_log1p_saturates_at_its_own_pole(self):
        # log1p's pole is at -1, not at 0.
        self.assertEqual(self._entry(aten.log1p.default, -1.0, 0.02, 0, 0.05, 0), -128)

    def test_log1p_continues_the_last_grid_point_before_its_pole(self):
        # Unlike the log family's pole at zero, -1 lands on the grid only for
        # some input scales. Where it does not, the undefined side continues an
        # ordinary finite entry rather than the rail.
        self.assertEqual(
            self._entry(aten.log1p.default, -1.2, 0.4, 0, 0.05, 0),
            self._entry(aten.log1p.default, -0.8, 0.4, 0, 0.05, 0),
        )

    def test_rsqrt_saturates_at_zero(self):
        self.assertEqual(self._entry(aten.rsqrt.default, 0.0, 0.02, -128, 0.05, 0), 127)

    def test_undefined_points_take_the_value_at_the_domain_boundary(self):
        # log of a negative is nan in eager, and the table cannot hold one. It
        # continues the pole instead of emitting the output zero point, which
        # would decode to 0.0 -- a value log can legitimately return, so a wrong
        # answer would be indistinguishable from a real one.
        self.assertEqual(self._entry(aten.log.default, -1.0, 0.02, 0, 0.05, 7), -128)
        self.assertEqual(self._entry(aten.rsqrt.default, -1.0, 0.02, 0, 0.05, 7), 127)
        # sqrt has no pole: it reaches zero at the boundary, so that is what the
        # undefined side continues.
        self.assertEqual(
            self._entry(aten.sqrt.default, -1.0, 0.02, 0, 0.05, 7),
            self._entry(aten.sqrt.default, 0.0, 0.02, 0, 0.05, 7),
        )

    def test_the_table_stays_monotone_across_the_pole(self):
        # The bug this replaces: an entry below the pole decoded to 0.0 and so
        # read larger than the entry just above it.
        lut = build_activation_lut(aten.log.default, 0.02, 0, 0.05, 7)
        self.assertEqual(list(lut), sorted(lut))

    def test_every_registry_lists_the_same_activations(self):
        """An activation needs a table, a quantizer pattern and a substitution.
        Missing the table raises, but missing either of the other two only makes
        the op quietly stay in float, which no end-to-end test would notice for
        an op nothing exercises yet."""
        # Functional and in-place spellings are matched separately by the
        # quantizer, so neither stands in for the other.
        quantizer = {
            _functional_edge_target(pattern[0])
            for pattern in ACTIVATION_OP_PATTERNS
            if len(pattern) == 1 and not pattern[0]._schema.name.endswith("_")
        }
        in_place = {
            _functional_edge_target(pattern[0])
            for pattern in ACTIVATION_OP_PATTERNS
            if len(pattern) == 1 and pattern[0]._schema.name.endswith("_")
        }
        # gelu is the exception: torch exposes no in-place spelling for it.
        self.assertEqual(in_place, quantizer - {aten.gelu.default})
        substituted = {
            target
            for target, fn in AtenToCortexMPass._DIALECT_SUBSTITUTIONS.items()
            if fn is _get_activation_replacement
        }
        self.assertEqual(quantizer, set(_ACTIVATION_FNS))
        self.assertEqual(quantizer, substituted)

    def test_in_domain_entries_match_eager(self):
        cases = (
            (aten.sqrt.default, torch.sqrt, 0.02, -128),
            (aten.rsqrt.default, torch.rsqrt, 0.02, -128),
            (aten.log.default, torch.log, 0.02, -128),
            (aten.log2.default, torch.log2, 0.02, -128),
            (aten.log10.default, torch.log10, 0.02, -128),
            (aten.log1p.default, torch.log1p, 0.02, 0),
            (aten.sigmoid.default, torch.sigmoid, 0.05, 0),
        )
        for target, eager, input_scale, input_zp in cases:
            with self.subTest(target=str(target)):
                output_scale, output_zp = 0.05, 3
                lut = build_activation_lut(
                    target, input_scale, input_zp, output_scale, output_zp
                )
                for q in range(-128, 128):
                    x = (q - input_zp) * input_scale
                    y = eager(torch.tensor(x, dtype=torch.float64)).item()
                    if not torch.isfinite(torch.tensor(y)):
                        continue
                    expected = max(
                        -128,
                        min(
                            127,
                            _round_half_away_from_zero(y / output_scale + output_zp),
                        ),
                    )
                    self.assertEqual(int(lut[q + 128]), expected, f"{target} at x={x}")


if __name__ == "__main__":
    unittest.main()
