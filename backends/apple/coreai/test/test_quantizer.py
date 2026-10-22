# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from executorch.backends.apple.coreai import (
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.backends.apple.coreai.quantizer.quantizer import CoreAIQuantizer
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_backend_modules,
)


def _coreai_quant_op_names(graph_module):
    """Names of call_function targets in a converted graph (test helper)."""
    return [
        getattr(node.target, "__name__", str(node.target))
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _model():
    return nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).eval()


class _ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _full_finalize(model, example_inputs):
    """coreai_opt standalone finalize(CoreAI) for equivalence checks."""
    from coreai_opt.common import ExportBackend
    from coreai_opt.quantization import Quantizer, QuantizerConfig

    q = Quantizer(model, QuantizerConfig())
    prepared = q.prepare(example_inputs)
    with q.calibration_mode():
        prepared(*example_inputs)
    return q.finalize(backend=ExportBackend.CoreAI)


def _constant_dtypes(graph_module):
    """dtype of every constant the converted graph holds, keyed by name.

    Targets can be dotted attribute paths (``0.bias``), which a plain getattr
    does not resolve, so walk them: a helper that silently returns nothing
    would make the "no such constant" assertions below pass vacuously.
    """
    dtypes = {}
    for node in graph_module.graph.nodes:
        if node.op != "get_attr":
            continue
        value = graph_module
        for part in str(node.target).split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if isinstance(value, torch.Tensor):
            dtypes[str(node.target)] = value.dtype
    return dtypes


def _weight_only_config(dtype, scale_dtype=None, granularity=None):
    """Weight-only config: state spec set, input and output specs left off.

    Weights live on the state spec, keyed by name; ``op_input_spec`` and
    ``op_output_spec`` take integer indices and cover activations, so omitting
    them is what makes a config weight-only.
    """
    from coreai_opt.quantization import (
        ModuleQuantizerConfig,
        QuantizationSpec,
        QuantizerConfig,
    )

    spec = QuantizationSpec(
        dtype=dtype,
        scale_dtype=scale_dtype,
        granularity=granularity or {"type": "per_channel", "axis": 0},
    )
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": spec},
            op_input_spec=None,
            op_output_spec=None,
        )
    )


@dataclass(frozen=True)
class FloatQuantCase:
    """One FP weight-quantization configuration and what it should produce."""

    name: str
    dtype: torch.dtype
    scale_dtype: Optional[torch.dtype]
    granularity: dict
    weight_dtype: torch.dtype
    scale_out_dtype: torch.dtype

    def config(self):
        return _weight_only_config(self.dtype, self.scale_dtype, self.granularity)


class CoreAIQuantizerTest(unittest.TestCase):
    def setUp(self):
        self.example_inputs = (torch.randn(2, 32),)

    def _convert(self, model):
        q = CoreAIQuantizer(model)
        prepared = q.prepare(self.example_inputs)
        with q.calibration_mode():
            prepared(*self.example_inputs)
        return q.convert()

    def test_convert_produces_coreai_quant_ops(self):
        names = set(_coreai_quant_op_names(self._convert(_model())))
        self.assertIn("quantize", names)
        self.assertIn("dequantize", names)
        self.assertIn("constexpr_blockwise_shift_scale", names)

    def test_convert_matches_full_finalize(self):
        # convert() IS the full coreai_opt finalize(CoreAI).
        converted = self._convert(_model())
        full = _full_finalize(_model(), self.example_inputs)
        self.assertEqual(
            Counter(_coreai_quant_op_names(converted)),
            Counter(_coreai_quant_op_names(full)),
        )

    def test_convert_output_is_exportable(self):
        # The key property: no fake-quant remains, so strict torch.export works.
        converted = self._convert(_model())
        ep = torch.export.export(converted, self.example_inputs)
        namespaces = {
            getattr(n.target, "namespace", None)
            for n in ep.graph.nodes
            if n.op == "call_function"
        }
        # Quant ops are lowered into the coreai namespace after export.
        self.assertIn("coreai", namespaces)

    def test_prepare_is_required_before_convert(self):
        q = CoreAIQuantizer(_model())
        with self.assertRaises(RuntimeError):
            q.convert()

    def test_prepare_is_required_before_the_mode_context_managers(self):
        """Otherwise coreai_opt's own error surfaces instead of ours."""
        for method in ("calibration_mode", "training_mode"):
            with self.subTest(method):
                q = CoreAIQuantizer(_model())
                with self.assertRaisesRegex(RuntimeError, f"before {method}"):
                    getattr(q, method)()


class CoreAIQuantizerConvBNTest(unittest.TestCase):
    def setUp(self):
        self.example_inputs = (torch.randn(1, 3, 16, 16),)

    def _convert(self, model):
        q = CoreAIQuantizer(model)
        prepared = q.prepare(self.example_inputs)
        with q.calibration_mode():
            prepared(*self.example_inputs)
        return q.convert()

    def test_convert_folds_conv_bn(self):
        # convert() folds conv+bn (pre-export) so no batch_norm survives.
        names = _coreai_quant_op_names(self._convert(_ConvBN().eval()))
        self.assertTrue(any("conv" in n for n in names), names)
        self.assertFalse(
            any("batch_norm" in n for n in names),
            f"batch_norm was not folded: {names}",
        )

    def test_convert_matches_full_finalize_conv_bn(self):
        converted = self._convert(_ConvBN().eval())
        full = _full_finalize(_ConvBN().eval(), self.example_inputs)
        self.assertEqual(
            Counter(_coreai_quant_op_names(converted)),
            Counter(_coreai_quant_op_names(full)),
        )


class CoreAIQuantizedLoweringTest(unittest.TestCase):
    """Full quantized lowering: quantize -> export -> to_edge_transform_and_lower."""

    def setUp(self):
        self.example_inputs = (torch.randn(2, 32),)

    def _lower(self):
        q = CoreAIQuantizer(_model())
        prepared = q.prepare(self.example_inputs)
        with q.calibration_mode():
            prepared(*self.example_inputs)
        ep = torch.export.export(q.convert(), self.example_inputs)
        return to_edge_transform_and_lower(ep, partitioner=[CoreAIPartitioner()])

    def test_quantized_linear_lowers_to_coreai(self):
        lowered = self._lower()
        gm = lowered.exported_program().graph_module

        delegates = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is executorch_call_delegate
        ]
        leftover = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is not executorch_call_delegate
            and "getitem" not in str(n.target)
        ]
        # The quantized subgraph (incl. coreai:: quant ops) is delegated and
        # converted by coreai-torch, leaving nothing outside the delegate.
        self.assertGreaterEqual(len(delegates), 1)
        self.assertEqual(
            leftover,
            [],
            f"unexpected ops left outside the delegate: "
            f"{[str(n.target) for n in leftover]}",
        )

    def test_quantized_lowers_through_to_executorch(self):
        lowered = self._lower()
        # A single CoreAI delegate, with its asset embedded (NDS present).
        lbms = get_lowered_backend_modules(lowered.exported_program().graph_module)
        self.assertEqual([lbm.backend_id for lbm in lbms], ["CoreAIBackend"])
        self.assertIsNotNone(lbms[0].named_data_store_output)
        # It lowers all the way to a non-empty .pte.
        buffer = bytes(lowered.to_executorch().buffer)
        self.assertGreater(len(buffer), 0)


@dataclass(frozen=True)
class IntWeightOnlyCase:
    """One integer weight-only configuration.

    ``preset`` names a ``QuantizerConfig.presets`` constructor where Apple
    ships one. int2 has none, so its specs are built to match the shape the
    presets use: signed, symmetric.
    """

    name: str
    preset: Optional[str]
    dtype: Optional[torch.dtype] = None
    granularity: Optional[dict] = None

    def config(self):
        if self.preset is not None:
            from coreai_opt.quantization import QuantizerConfig

            return getattr(QuantizerConfig.presets, self.preset)()
        return _weight_only_config(self.dtype, granularity=self.granularity)


class IntegerWeightOnlyQuantizationTest(unittest.TestCase):
    """Signed integer weight-only quantization across widths.

    Signed symmetric is the sanctioned sub-byte path: every preset Apple ships
    (``w8``, ``w4``, ``w4_per_block``) is signed and symmetric, and there is no
    unsigned or asymmetric preset. The unsigned and affine variants do lower,
    but they are not what the presets select.

    int2 is covered here because it lowers cleanly, but note there is no int2
    preset at all, in either granularity.

    The default config quantizes activations as well, so these cover a
    configuration the tests above do not reach.
    """

    CASES = (
        IntWeightOnlyCase(name="int8", preset="w8"),
        IntWeightOnlyCase(name="int4", preset="w4"),
        IntWeightOnlyCase(name="int4_per_block", preset="w4_per_block"),
        IntWeightOnlyCase(name="int2", preset=None, dtype=torch.int2),
        IntWeightOnlyCase(
            # Per-block is the more sensible choice at 2 bits: more scales to
            # compensate for a four-level grid.
            name="int2_per_block",
            preset=None,
            dtype=torch.int2,
            granularity={"type": "per_block", "block_size": (1, 16)},
        ),
    )

    def setUp(self):
        self.example_inputs = (torch.randn(2, 32),)

    def _convert(self, config):
        quantizer = CoreAIQuantizer(_model(), config)
        prepared = quantizer.prepare(self.example_inputs)
        with quantizer.calibration_mode():
            prepared(*self.example_inputs)
        return quantizer.convert()

    def _pte_size(self, config):
        ep = torch.export.export(self._convert(config), self.example_inputs)
        lowered = to_edge_transform_and_lower(
            ep,
            transform_passes=get_default_passes(),
            partitioner=[CoreAIPartitioner()],
            compile_config=get_default_compile_config(),
        )
        return len(bytes(lowered.to_executorch().buffer))

    def test_activations_are_left_unquantized(self):
        """The property that makes these weight-only, rather than just narrow."""
        for case in self.CASES:
            with self.subTest(case.name):
                converted = self._convert(case.config())
                ops = Counter(_coreai_quant_op_names(converted))
                self.assertEqual(ops["quantize"], 0)
                self.assertEqual(ops["dequantize"], 0)
                self.assertGreater(ops["constexpr_blockwise_shift_scale"], 0)

                constants = _constant_dtypes(converted)
                self.assertEqual(
                    [n for n in constants if n.startswith("activation_post_process")],
                    [],
                )

    def test_default_config_quantizes_activations(self):
        """Contrast: the default config does insert activation quant.

        Without this the weight-only assertions could pass against a build
        that had stopped quantizing activations for everyone.
        """
        ops = Counter(_coreai_quant_op_names(self._convert(None)))
        self.assertGreater(ops["quantize"], 0)
        self.assertGreater(ops["dequantize"], 0)

    def test_each_width_lowers_through_to_executorch(self):
        for case in self.CASES:
            with self.subTest(case.name):
                self.assertGreater(self._pte_size(case.config()), 0)

    def test_smaller_blocks_add_scales(self):
        """Per-block at 2 bits trades size for accuracy, so it must cost more.

        Confirms the granularity is applied rather than silently ignored: the
        block-16 variant carries more scale constants than per-channel.
        """
        by_name = {case.name: case for case in self.CASES}
        per_channel = self._pte_size(by_name["int2"].config())
        per_block = self._pte_size(by_name["int2_per_block"].config())
        self.assertGreater(
            per_block,
            per_channel,
            f"block-16 int2 ({per_block}) should exceed per-channel "
            f"({per_channel}); equal sizes suggest granularity was ignored",
        )

    def test_narrower_weights_produce_smaller_programs(self):
        """Guards a silent fallback that would keep the wider dtype.

        Sub-byte weights surface as int8 in the fx graph (the byte container,
        packed later), so the asset size is the honest observable rather than
        the constant's dtype.
        """
        sizes = {
            case.name: self._pte_size(case.config())
            for case in self.CASES
            if case.name in ("int2", "int4", "int8")
        }
        self.assertLess(sizes["int2"], sizes["int4"], sizes)
        self.assertLess(sizes["int4"], sizes["int8"], sizes)


class FloatQuantizationTest(unittest.TestCase):
    """FP8 and FP4 weight quantization, end to end.

    coreai_opt expresses these as ordinary ``torch.dtype`` values on a
    ``QuantizationSpec``, with ``scale_dtype`` selecting the scale form: an
    fp32 scale for plain FP8, or a power-of-two ``float8_e8m0fnu`` scale for
    OCP Microscaling (FP4 always, FP8 optionally).
    """

    CASES = (
        FloatQuantCase(
            name="fp8_e4m3_per_channel",
            dtype=torch.float8_e4m3fn,
            scale_dtype=None,
            granularity={"type": "per_channel", "axis": 0},
            weight_dtype=torch.float8_e4m3fn,
            scale_out_dtype=torch.float32,
        ),
        FloatQuantCase(
            name="fp8_e5m2_per_channel",
            dtype=torch.float8_e5m2,
            scale_dtype=None,
            granularity={"type": "per_channel", "axis": 0},
            weight_dtype=torch.float8_e5m2,
            scale_out_dtype=torch.float32,
        ),
        FloatQuantCase(
            name="fp8_e4m3_microscaling",
            dtype=torch.float8_e4m3fn,
            scale_dtype=torch.float8_e8m0fnu,
            granularity={"type": "per_block", "block_size": (1, 32)},
            weight_dtype=torch.float8_e4m3fn,
            scale_out_dtype=torch.float8_e8m0fnu,
        ),
        FloatQuantCase(
            # FP4 is packed two values per byte, so the quantized weight
            # surfaces as uint8; coreai-torch recovers the logical shape.
            name="fp4_e2m1_microscaling",
            dtype=torch.float4_e2m1fn_x2,
            scale_dtype=None,
            granularity={"type": "per_block", "block_size": (1, 32)},
            weight_dtype=torch.uint8,
            scale_out_dtype=torch.float8_e8m0fnu,
        ),
    )

    def setUp(self):
        self.example_inputs = (torch.randn(2, 32),)

    def _convert(self, case):
        quantizer = CoreAIQuantizer(_model(), case.config())
        prepared = quantizer.prepare(self.example_inputs)
        with quantizer.calibration_mode():
            prepared(*self.example_inputs)
        return quantizer.convert()

    def _pte_size(self, case):
        converted = self._convert(case)
        ep = torch.export.export(converted, self.example_inputs)
        lowered = to_edge_transform_and_lower(
            ep,
            transform_passes=get_default_passes(),
            partitioner=[CoreAIPartitioner()],
            compile_config=get_default_compile_config(),
        )
        return len(bytes(lowered.to_executorch().buffer))

    def test_quantized_weight_and_scale_dtypes(self):
        """The requested dtype must reach the graph, not silently degrade."""
        for case in self.CASES:
            with self.subTest(case.name):
                constants = _constant_dtypes(self._convert(case))
                weights = [
                    dtype
                    for name, dtype in constants.items()
                    if name.endswith("weight_quantized")
                ]
                scales = [
                    dtype
                    for name, dtype in constants.items()
                    if name.endswith("weight_scale")
                ]
                self.assertTrue(weights, f"no quantized weight in {constants}")
                self.assertEqual(set(weights), {case.weight_dtype})
                self.assertEqual(set(scales), {case.scale_out_dtype})

    def test_float_quantization_has_no_zero_point(self):
        """FP formats are symmetric only, so zero-point is always implicit."""
        for case in self.CASES:
            with self.subTest(case.name):
                constants = _constant_dtypes(self._convert(case))
                self.assertTrue(constants, "no constants collected")
                self.assertEqual([n for n in constants if n.endswith("zero_point")], [])

    def test_lowers_through_to_executorch(self):
        for case in self.CASES:
            with self.subTest(case.name):
                self.assertGreater(self._pte_size(case), 0)

    def test_fp4_is_smaller_than_fp8(self):
        """Guards against a silent fallback that would keep the wider dtype.

        Both sides use the same per-block granularity and e8m0 scales, so the
        difference is the weight dtype alone.
        """
        by_name = {case.name: case for case in self.CASES}
        fp4 = self._pte_size(by_name["fp4_e2m1_microscaling"])
        fp8 = self._pte_size(by_name["fp8_e4m3_microscaling"])
        self.assertLess(
            fp4,
            fp8,
            f"fp4 pte ({fp4}) should be smaller than fp8 at the same "
            f"granularity ({fp8}); equal sizes suggest the requested dtype was "
            f"not applied",
        )


if __name__ == "__main__":
    unittest.main()
