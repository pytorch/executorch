# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product
from unittest.mock import patch

import executorch.exir as exir
import torch
import torch.nn as nn
from executorch.backends.transforms.fuse_rms_norm import (
    _frozen_rmsnorm_weight,
    FuseRMSNormPass,
)
from executorch.backends.transforms.utils import (
    _validate_graph_signature,
    create_constant_placeholder,
)
from executorch.exir import EdgeCompileConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase
from executorch.exir.pass_manager import ExportedProgramPassManager
from executorch.exir.passes.remove_graph_asserts_pass import RemoveGraphAssertsPass
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import export
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_base import PassResult


def _to_edge_ep(module, inputs, dynamic_shapes=None):
    ep = export(module, inputs, dynamic_shapes=dynamic_shapes, strict=False)
    # Functionalize writes so mutable buffers are represented in the signature.
    ep = ep.run_decompositions({})
    return exir.to_edge(
        ep,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    ).exported_program()


def _export_ep(module, inputs, edge=False, dynamic_shapes=None):
    if edge:
        return _to_edge_ep(module, inputs, dynamic_shapes)
    return export(module, inputs, dynamic_shapes=dynamic_shapes, strict=False)


def _find_nodes(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _rmsnorm_input(dtype, seq=4):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, seq, 16, generator=generator)
    # Include epsilon-dominated rows and values whose FP16 square overflows.
    x[0, 0] *= 1e-4
    x[1, -1] *= 1000
    return x.to(dtype)


class _FP32RMSNorm(nn.Module):
    """Expose the FP32 result so output rounding cannot hide regressions."""

    def __init__(
        self, with_scale=True, use_rsqrt=False, commute_core=False, commute_scale=False
    ):
        super().__init__()
        self.weight = (
            nn.Parameter(torch.linspace(-1.7, 2.3, 16)) if with_scale else None
        )
        self.use_rsqrt = use_rsqrt
        self.commute_core = commute_core
        self.commute_scale = commute_scale

    def forward(self, x):
        f32 = x.float()
        variance = f32.pow(2).mean(-1, keepdim=True) + 1e-6
        inv = variance.rsqrt() if self.use_rsqrt else variance.pow(-0.5)
        out = inv * f32 if self.commute_core else f32 * inv
        if self.weight is not None:
            weight = self.weight.float()
            out = weight * out if self.commute_scale else out * weight
        return out.to(x.dtype), out


class _CastRMSNorm(nn.Module):
    """Single-output norm with explicit export-stable casts and optional fanout."""

    def __init__(self, with_scale=True, escape=None, output_dtype=None):
        super().__init__()
        self.weight = (
            nn.Parameter(torch.linspace(-1.7, 2.3, 16)) if with_scale else None
        )
        self.escape = escape
        self.output_dtype = output_dtype

    def forward(self, x):
        f32 = torch.ops.aten._to_copy.default(x, dtype=torch.float32)
        out = f32 * (f32.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.5)
        weight = None
        if self.weight is not None:
            weight = torch.ops.aten._to_copy.default(self.weight, dtype=torch.float32)
            out = out * weight
        result = torch.ops.aten._to_copy.default(
            out, dtype=self.output_dtype or x.dtype
        )
        if self.escape is not None:
            extra = {"input": f32, "weight": weight, "output": out}[self.escape]
            return result, extra
        return result


class _OffsetRMSNorm(nn.Module):
    """Gemma-style FP32 effective weight, with optional shared intermediates."""

    def __init__(self, commute=False, escape=None):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-0.7, 1.3, 16))
        self.commute = commute
        self.escape = escape

    def forward(self, x):
        f32 = torch.ops.aten._to_copy.default(x, dtype=torch.float32)
        weight = torch.ops.aten._to_copy.default(self.weight, dtype=torch.float32)
        gamma = 1.0 + weight if self.commute else weight + 1.0
        norm = f32 * (f32.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.5)
        scaled = norm * gamma
        result = torch.ops.aten._to_copy.default(scaled, dtype=x.dtype)
        if self.escape is not None:
            extra = {
                "input": f32,
                "gamma": gamma,
                "scaled": scaled,
                "weight": self.weight,
            }[self.escape]
            return result, extra
        return result


class _LossyWeightRMSNorm(nn.Module):
    def __init__(
        self, dtype=torch.float32, *, source="parameter", expression="add", shared=False
    ):
        super().__init__()
        weight = torch.linspace(-1.3, 1.7, 16).to(dtype)
        if source == "parameter":
            self.weight = nn.Parameter(weight)
        elif source in ("buffer", "mutable"):
            self.register_buffer("weight", weight)
        else:
            self.weight = weight
        self.one = torch.tensor(1.0)
        self.source = source
        self.expression = expression
        self.shared = shared

    def forward(self, x, weight=None):
        gamma = self.weight if weight is None else weight
        gamma = gamma.float()
        if self.expression == "add":
            gamma = gamma + 1.0
        elif self.expression == "lifted_left":
            gamma = self.one + gamma
        elif self.expression == "lifted_right":
            gamma = gamma + self.one
        elif self.expression == "unknown":
            gamma = gamma.sin()
        elif self.expression == "alpha":
            gamma = torch.add(gamma, 1.0, alpha=2)
        f32 = x.float()
        normalized = f32 * torch.rsqrt(f32.pow(2).mean(-1, keepdim=True) + 1e-6)
        result = (normalized * gamma).to(x.dtype)
        if self.source == "mutable":
            self.weight.add_(0.125)
        return (result, self.weight.sin()) if self.shared else result


class _RMSNormAssertions:
    def _assert_signature(self, ep):
        ep.graph_module.graph.lint()
        _validate_graph_signature(ep)
        self.assertEqual(
            [node.name for node in ep.graph.nodes if node.op == "placeholder"],
            [spec.arg.name for spec in ep.graph_signature.input_specs],
        )
        outputs = next(node for node in ep.graph.nodes if node.op == "output")
        self.assertEqual(
            [node.name for node in outputs.args[0]],
            [spec.arg.name for spec in ep.graph_signature.output_specs],
        )

    def _prepared_weight(self, ep):
        norms = _find_nodes(ep.graph_module, torch.ops.aten.rms_norm.default)
        self.assertEqual(len(norms), 1)
        node = norms[0].args[2]
        self.assertEqual(node.op, "placeholder")
        target = ep.graph_signature.inputs_to_lifted_tensor_constants[node.name]
        return norms[0], ep.constants[target]

    def _assert_fused(self, ep, with_scale, compute_dtype=torch.float32):
        self._assert_signature(ep)
        gm = ep.graph_module
        nodes = _find_nodes(gm, torch.ops.aten.rms_norm.default)
        self.assertEqual(len(nodes), 1)
        fused = nodes[0]
        self.assertEqual(fused.args[1], [16])
        self.assertEqual(fused.args[2] is not None, with_scale)
        self.assertAlmostEqual(fused.args[3], 1e-6)
        self.assertEqual(fused.args[0].meta["val"].dtype, compute_dtype)
        self.assertEqual(fused.meta["val"].dtype, compute_dtype)
        if with_scale:
            self.assertEqual(fused.args[2].meta["val"].dtype, compute_dtype)
        for aten in (torch.ops.aten, exir_ops.edge.aten):
            for target in (aten.pow.Tensor_Scalar, aten.mean.dim, aten.rsqrt.default):
                self.assertEqual(_find_nodes(gm, target), [])
        return fused

    def _assert_no_casts(self, ep):
        for aten in (torch.ops.aten, exir_ops.edge.aten):
            for target in (aten._to_copy.default, aten.to.dtype, aten.type_as.default):
                self.assertEqual(_find_nodes(ep.graph_module, target), [])

    def _assert_close(self, actual, expected, tolerance_dtype=None):
        tolerances = {
            torch.float32: (2e-6, 2e-6),
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (8e-3, 8e-3),
        }
        rtol, atol = tolerances[tolerance_dtype or expected.dtype]
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def _apply(self, ep, **options):
        result = FuseRMSNormPass(**options)(ep)
        self.assertTrue(result.modified)
        self.assertIsInstance(result.exported_program, torch.export.ExportedProgram)
        self._assert_signature(result.exported_program)
        return result.exported_program

    def _assert_noop(self, ep, transform=None):
        transform = FuseRMSNormPass() if transform is None else transform
        before = (str(ep.graph), str(ep.graph_signature), tuple(ep.constants))
        result = transform(ep)
        self.assertFalse(result.modified)
        self.assertIs(result.exported_program, ep)
        self.assertEqual(
            (str(ep.graph), str(ep.graph_signature), tuple(ep.constants)), before
        )
        self._assert_signature(ep)


class TestFuseRMSNormPass(_RMSNormAssertions, unittest.TestCase):
    """Verify matching, rounding policies, and ExportedProgram invariants."""

    def test_default_preserves_casts_and_opt_in_folds_fresh_graphs(self):
        """Compare fresh raw and Edge graphs under both cast-folding policies."""
        self.assertFalse(FuseRMSNormPass().fold_dtype_casts)
        self.assertFalse(FuseRMSNormPass().allow_lossy_weight_casts)
        for dtype, with_scale, edge, fold in product(
            (torch.float16, torch.bfloat16), (False, True), (False, True), (False, True)
        ):
            with self.subTest(dtype=dtype, scale=with_scale, edge=edge, fold=fold):
                model = _CastRMSNorm(with_scale=with_scale).to(dtype).eval()
                x = _rmsnorm_input(dtype)
                ep = _export_ep(model, (x,), edge)
                aten = exir_ops.edge.aten if edge else torch.ops.aten
                casts = _find_nodes(ep.graph_module, aten._to_copy.default)
                self.assertEqual(len(casts), 3 if with_scale else 2)
                ep = self._apply(ep, fold_dtype_casts=fold)
                fused = self._assert_fused(
                    ep, with_scale, compute_dtype=dtype if fold else torch.float32
                )
                if fold:
                    self.assertEqual(fused.args[0].op, "placeholder")
                    self._assert_no_casts(ep)
                else:
                    self.assertEqual(
                        _find_nodes(ep.graph_module, aten._to_copy.default), casts
                    )
                    self.assertIs(fused.args[0], casts[0])
                    self.assertIs(casts[-1].args[0], fused)
                self._assert_close(ep.module()(x), model(x))

    def test_idempotence(self):
        """Leave fused graphs, signatures, and storage unchanged on a second call."""
        for with_scale, edge, fold in product((False, True), repeat=3):
            with self.subTest(scale=with_scale, edge=edge, fold=fold):
                model = _CastRMSNorm(with_scale=with_scale).to(torch.float16).eval()
                x = _rmsnorm_input(torch.float16)
                ep = self._apply(_export_ep(model, (x,), edge), fold_dtype_casts=fold)
                self._assert_fused(
                    ep,
                    with_scale,
                    compute_dtype=torch.float16 if fold else torch.float32,
                )
                self._assert_noop(ep, FuseRMSNormPass(fold_dtype_casts=fold))
                self._assert_close(ep.module()(x), model(x))

    def test_casts_absorbed_with_matching_device_and_layout(self):
        """Accept explicit device and layout kwargs that do not convert storage."""
        for dtype, with_scale in product(
            (torch.float16, torch.bfloat16), (False, True)
        ):
            with self.subTest(dtype=dtype, scale=with_scale):
                model = _CastRMSNorm(with_scale=with_scale).to(dtype).eval()
                x = _rmsnorm_input(dtype)
                ep = _export_ep(model, (x,))
                for cast in _find_nodes(
                    ep.graph_module, torch.ops.aten._to_copy.default
                ):
                    cast.kwargs = dict(
                        cast.kwargs, device=torch.device("cpu"), layout=torch.strided
                    )
                ep = self._apply(ep, fold_dtype_casts=True)
                self._assert_fused(ep, with_scale, compute_dtype=dtype)
                self._assert_no_casts(ep)
                self._assert_close(ep.module()(x), model(x))

    def test_casts_preserved_for_shared_fp32_values(self):
        """Keep boundaries when FP32 activations, weights, or outputs escape."""
        for escape in ("input", "weight", "output"):
            with self.subTest(escape=escape):
                model = _CastRMSNorm(escape=escape).to(torch.bfloat16).eval()
                x = _rmsnorm_input(torch.bfloat16)
                ep = _export_ep(model, (x,))
                casts = _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default)
                ep = self._apply(ep, fold_dtype_casts=True)
                fused = self._assert_fused(ep, with_scale=True)
                self.assertIs(fused.args[0], casts[0])
                self.assertEqual(
                    _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default), casts
                )
                actual, escaped = ep.module()(x)
                expected, expected_escaped = model(x)
                self._assert_close(actual, expected)
                self._assert_close(escaped, expected_escaped)

    def test_casts_preserved_for_dtype_mismatch(self):
        """Do not absorb mismatched input, weight, or output dtypes."""
        for output_dtype, weight_dtype in (
            (torch.bfloat16, torch.float16),
            (torch.float16, torch.bfloat16),
            (torch.float16, torch.float32),
        ):
            with self.subTest(output_dtype=output_dtype, weight_dtype=weight_dtype):
                model = _CastRMSNorm(output_dtype=output_dtype).to(weight_dtype).eval()
                x = _rmsnorm_input(torch.float16)
                ep = _export_ep(model, (x,))
                casts = _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default)
                ep = self._apply(ep, fold_dtype_casts=True)
                self._assert_fused(ep, with_scale=True)
                self.assertEqual(
                    _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default), casts
                )
                self._assert_close(ep.module()(x), model(x))

    def test_casts_preserved_for_non_dtype_conversions(self):
        """Reject device, layout, and memory-format conversion boundaries."""
        for position, (kwarg, value) in product(
            range(3),
            (
                ("device", torch.device("meta")),
                ("layout", torch.sparse_coo),
                ("memory_format", torch.channels_last),
            ),
        ):
            with self.subTest(position=position, kwarg=kwarg):
                model = _CastRMSNorm().to(torch.float16).eval()
                ep = _export_ep(model, (_rmsnorm_input(torch.float16),))
                casts = _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default)
                # Exercise guards without a second device or sparse RMSNorm kernel.
                # This deliberately synthetic graph is not executed.
                casts[position].kwargs = dict(casts[position].kwargs, **{kwarg: value})
                ep = self._apply(ep, fold_dtype_casts=True)
                self._assert_fused(ep, with_scale=True)
                self.assertEqual(
                    _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default), casts
                )
                self.assertEqual(casts[position].kwargs[kwarg], value)

    def test_fp32_result_and_dynamic_sequence(self):
        """Preserve exposed FP32 results across dynamic sequence lengths."""
        for dtype, with_scale in product(
            (torch.float32, torch.float16, torch.bfloat16), (False, True)
        ):
            with self.subTest(dtype=dtype, with_scale=with_scale):
                model = _FP32RMSNorm(with_scale=with_scale).to(dtype).eval()
                ep = _export_ep(
                    model,
                    (_rmsnorm_input(dtype),),
                    dynamic_shapes={"x": {1: torch.export.Dim("seq", min=1, max=8)}},
                )
                ep = self._apply(ep, fold_dtype_casts=True)
                self._assert_fused(ep, with_scale)
                self.assertEqual(
                    _find_nodes(ep.graph_module, torch.ops.aten.mul.Tensor), []
                )
                for seq in (1, 4, 7):
                    x = _rmsnorm_input(dtype, seq)
                    actual, actual_f32 = ep.module()(x)
                    expected, expected_f32 = model(x)
                    self._assert_close(actual, expected)
                    self._assert_close(actual_f32, expected_f32)

    def test_weighted_rsqrt_and_commuted_multiplications(self):
        """Match both inverse-root forms and either multiplication order."""
        for use_rsqrt, commute_core, commute_scale in product((False, True), repeat=3):
            with self.subTest(
                use_rsqrt=use_rsqrt, core=commute_core, scale=commute_scale
            ):
                model = _FP32RMSNorm(
                    use_rsqrt=use_rsqrt,
                    commute_core=commute_core,
                    commute_scale=commute_scale,
                ).eval()
                x = _rmsnorm_input(torch.float32)
                ep = self._apply(_export_ep(model, (x,)))
                self._assert_fused(ep, with_scale=True)
                self.assertEqual(
                    _find_nodes(ep.graph_module, torch.ops.aten.mul.Tensor), []
                )
                self._assert_close(ep.module()(x)[1], model(x)[1])

    def test_rounding_before_external_scale_is_preserved(self):
        """Keep scaling outside a norm whose result is rounded before scaling."""

        class CastBeforeScale(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.linspace(-1.7, 2.3, 16))

            def forward(self, x):
                f32 = x.float()
                normalized = f32 * (f32.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
                return self.weight * normalized.to(x.dtype)

        for dtype, fold in product((torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, fold=fold):
                model = CastBeforeScale().eval()
                x = _rmsnorm_input(dtype)
                expected = model(x)
                ep = self._apply(_to_edge_ep(model, (x,)), fold_dtype_casts=fold)
                fused = self._assert_fused(ep, False, dtype if fold else torch.float32)
                muls = _find_nodes(ep.graph_module, exir_ops.edge.aten.mul.Tensor)
                self.assertEqual(len(muls), 1)
                casts = _find_nodes(
                    ep.graph_module, exir_ops.edge.aten._to_copy.default
                )
                self.assertEqual(len(casts), 1 if fold else 3)
                if fold:
                    self.assertIs(casts[0].args[0], fused)
                else:
                    self.assertIs(casts[-2].args[0], fused)
                    self.assertEqual(casts[-2].kwargs["dtype"], dtype)
                self.assertEqual(casts[-1].kwargs["dtype"], torch.float32)
                self.assertIn(casts[-1], muls[0].args)
                self.assertEqual(muls[0].meta["val"].dtype, torch.float32)
                self._assert_close(ep.module()(x), expected, dtype if fold else None)
                premature_scale = (
                    torch.nn.functional.rms_norm(x.float(), (16,), model.weight, 1e-6)
                    .to(dtype)
                    .float()
                )
                self.assertGreater(
                    (expected - premature_scale).abs().max().item(), 1e-4
                )

    def test_near_match_semantics_are_not_fused(self):
        """Reject altered powers, reductions, epsilon values, and compute dtypes."""

        class WeightedNorm(nn.Module):
            def __init__(self, normalize):
                super().__init__()
                self.normalize = normalize
                self.weight = nn.Parameter(torch.linspace(-1.7, 2.3, 16))

            def forward(self, *inputs):
                return self.weight * self.normalize(*inputs)

        # Square dimensions keep the incorrect reductions broadcastable.
        x = torch.linspace(0.1, 2.5, 256).reshape(16, 16)
        cases = (
            (
                "square",
                lambda x: x * (x.pow(3).mean(-1, keepdim=True) + 1e-6).rsqrt(),
                (x,),
            ),
            (
                "inverse",
                lambda x: x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.4),
                (x,),
            ),
            (
                "axis",
                lambda x: x * (x.pow(2).mean(0, keepdim=True) + 1e-6).rsqrt(),
                (x,),
            ),
            (
                "axes",
                lambda x: x * (x.pow(2).mean([0, 1], keepdim=True) + 1e-6).rsqrt(),
                (x,),
            ),
            (
                "keepdim",
                lambda x: x * (x.pow(2).mean(-1, keepdim=False) + 1e-6).rsqrt(),
                (x,),
            ),
            (
                "input",
                lambda x, other: other
                * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt(),
                (x, x.flip(0)),
            ),
            (
                "tensor_epsilon",
                lambda x, eps: x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt(),
                (x, torch.tensor(1e-6)),
            ),
            (
                "vector_epsilon",
                lambda x, eps: x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt(),
                (x, torch.full((16,), 1e-6)),
            ),
            (
                "negative_epsilon",
                lambda x: x * (x.pow(2).mean(-1, keepdim=True) + (-1e-6)).rsqrt(),
                (x,),
            ),
            (
                "infinite_epsilon",
                lambda x: x * (x.pow(2).mean(-1, keepdim=True) + float("inf")).rsqrt(),
                (x,),
            ),
            (
                "alpha",
                lambda x: x
                * torch.add(x.pow(2).mean(-1, keepdim=True), 1e-6, alpha=2).rsqrt(),
                (x,),
            ),
            (
                "mean_dtype",
                lambda x: x
                * (x.pow(2).mean(-1, keepdim=True, dtype=torch.float64) + 1e-6).rsqrt(),
                (x,),
            ),
            (
                "inverse_dtype",
                lambda x: x
                * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt().to(torch.float16),
                (x,),
            ),
        )
        for case, normalize, inputs in cases:
            with self.subTest(case=case):
                model = WeightedNorm(normalize).eval()
                ep = _export_ep(model, inputs)
                self._assert_noop(ep)
                self.assertEqual(
                    _find_nodes(ep.graph_module, torch.ops.aten.rms_norm.default), []
                )
                torch.testing.assert_close(
                    ep.module()(*inputs), model(*inputs), rtol=2e-6, atol=2e-6
                )

    def test_symbolic_hidden_width_is_not_fused(self):
        """Require a static normalized width without restricting leading axes."""
        model = _FP32RMSNorm(with_scale=False).eval()
        ep = _export_ep(
            model,
            (_rmsnorm_input(torch.float32),),
            dynamic_shapes={"x": {2: torch.export.Dim("width", min=4, max=32)}},
        )
        self._assert_noop(ep)
        for width in (8, 24):
            x = torch.linspace(-2.0, 2.0, 8 * width).reshape(2, 4, width)
            torch.testing.assert_close(ep.module()(x), model(x))

    def test_non_fp32_core_is_not_fused(self):
        """Do not substitute RMSNorm for native low-precision or FP64 arithmetic."""

        class NativeDtypeNorm(nn.Module):
            def forward(self, x):
                return x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()

        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.linspace(0.1, 2.5, 32).reshape(2, 16).to(dtype)
                ep = _export_ep(NativeDtypeNorm(), (x,))
                self._assert_noop(ep)
                self.assertEqual(
                    _find_nodes(ep.graph_module, torch.ops.aten.rms_norm.default), []
                )

    def test_intermediate_fanout_is_not_fused(self):
        """Preserve each escaping intermediate of the decomposition."""

        class Fanout(nn.Module):
            def __init__(self, stage):
                super().__init__()
                self.stage = stage
                self.weight = nn.Parameter(torch.linspace(-1.7, 2.3, 16))

            def forward(self, x):
                square = x.pow(2)
                mean = square.mean(-1, keepdim=True)
                added = mean + 1e-6
                inverse = added.pow(-0.5)
                normalized = x * inverse
                return (
                    normalized * self.weight,
                    (square, mean, added, inverse)[self.stage],
                )

        x = _rmsnorm_input(torch.float32)
        for stage in range(4):
            with self.subTest(stage=stage):
                model = Fanout(stage).eval()
                ep = _export_ep(model, (x,))
                self._assert_noop(ep)
                torch.testing.assert_close(
                    ep.module()(x), model(x), rtol=2e-6, atol=2e-6
                )

    def test_unsafe_scale_fuses_only_core(self):
        """Keep unsupported scale shapes and dtypes outside the fused core."""

        class ScaledNorm(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = nn.Parameter(weight)

            def forward(self, x):
                normalized = x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
                return normalized * self.weight

        weights = (
            torch.tensor(1.3),
            torch.tensor([1.3]),
            torch.linspace(-1.7, 2.3, 16).reshape(1, 16),
            torch.linspace(-1.7, 2.3, 16).reshape(16, 1),
            torch.linspace(-1.7, 2.3, 16).to(torch.float16),
            torch.linspace(-1.7, 2.3, 16).to(torch.bfloat16),
            torch.linspace(-1.7, 2.3, 16).to(torch.float64),
        )
        x = _rmsnorm_input(torch.float32, seq=16)
        for weight in weights:
            with self.subTest(shape=tuple(weight.shape), dtype=weight.dtype):
                model = ScaledNorm(weight).eval()
                ep = self._apply(_export_ep(model, (x,)))
                fused = self._assert_fused(ep, with_scale=False)
                muls = _find_nodes(ep.graph_module, torch.ops.aten.mul.Tensor)
                self.assertEqual(len(muls), 1)
                self.assertIn(fused, muls[0].args)
                torch.testing.assert_close(
                    ep.module()(x), model(x), rtol=2e-6, atol=2e-6
                )

    def test_normalized_output_fanout_keeps_external_scale(self):
        """Preserve both normalized and externally scaled output consumers."""

        class Fanout(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.linspace(-1.7, 2.3, 16))

            def forward(self, x):
                f32 = x.float()
                normalized = f32 * (f32.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.5)
                return normalized, normalized * self.weight, f32

        model = Fanout().eval()
        x = _rmsnorm_input(torch.bfloat16)
        ep = self._apply(_export_ep(model, (x,)))
        fused = self._assert_fused(ep, with_scale=False)
        self.assertEqual(len(fused.users), 2)
        self.assertEqual(
            len(_find_nodes(ep.graph_module, torch.ops.aten.mul.Tensor)), 1
        )
        torch.testing.assert_close(ep.module()(x), model(x), rtol=2e-6, atol=2e-6)

    def test_weighted_rsqrt_edge_graph(self):
        """Match a weighted rsqrt decomposition in the Edge dialect."""
        model = _FP32RMSNorm(use_rsqrt=True).eval()
        x = _rmsnorm_input(torch.float32)
        ep = _to_edge_ep(model, (x,))
        self.assertEqual(
            len(_find_nodes(ep.graph_module, exir_ops.edge.aten.rsqrt.default)), 1
        )
        ep = self._apply(ep)
        self._assert_fused(ep, with_scale=True)
        torch.testing.assert_close(ep.module()(x), model(x), rtol=2e-6, atol=2e-6)

    def test_lossy_weight_casts_require_boundary_folding(self):
        """Reject an opt-in that cannot fold the activation boundaries."""
        with self.assertRaisesRegex(ValueError, "requires fold_dtype_casts=True"):
            FuseRMSNormPass(allow_lossy_weight_casts=True)

    def test_offset_weight_rounding_is_opt_in_and_after_fp32_add(self):
        """Materialize FP32 effective weights only under explicit opt-in."""
        for dtype, edge, commute, allow in product(
            (torch.float16, torch.bfloat16), (False, True), (False, True), (False, True)
        ):
            with self.subTest(dtype=dtype, edge=edge, commute=commute, allow=allow):
                model = _OffsetRMSNorm(commute=commute).to(dtype).eval()
                x = _rmsnorm_input(dtype)
                ep = _export_ep(model, (x,), edge)
                aten = exir_ops.edge.aten if edge else torch.ops.aten
                casts = _find_nodes(ep.graph_module, aten._to_copy.default)
                ep = self._apply(
                    ep, fold_dtype_casts=True, allow_lossy_weight_casts=allow
                )
                fused = self._assert_fused(ep, True, dtype if allow else torch.float32)
                if allow:
                    self.assertEqual(fused.args[0].op, "placeholder")
                    _, weight = self._prepared_weight(ep)
                    self.assertEqual(fused.args[2].name, "_rmsnorm_weight_0")
                    self.assertEqual(tuple(weight.shape), (16,))
                    self.assertEqual(weight.dtype, dtype)
                    self._assert_no_casts(ep)
                    self.assertEqual(_find_nodes(ep.graph_module, aten.add.Tensor), [])
                    expected_weight = (model.weight.float() + 1.0).to(dtype)
                    torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)
                    expected = torch.rms_norm(x, [16], expected_weight, 1e-6)
                    torch.testing.assert_close(ep.module()(x), expected, rtol=0, atol=0)
                else:
                    self.assertEqual(
                        _find_nodes(ep.graph_module, aten._to_copy.default), casts
                    )
                self._assert_close(ep.module()(x), model(x))
                self._assert_noop(
                    ep,
                    FuseRMSNormPass(
                        fold_dtype_casts=True, allow_lossy_weight_casts=allow
                    ),
                )

    def test_runtime_input_weight_add_precedes_downcast(self):
        """Retain a live FP32 add followed by rounding for changing input weights."""
        for dtype, edge in product((torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, edge=edge):
                model = _LossyWeightRMSNorm()
                x = _rmsnorm_input(dtype)
                weight = torch.linspace(-1.0002, -0.9998, 16)
                ep = _export_ep(model, (x, weight), edge)
                # Unlifting used to remove these assertion users implicitly. Keep
                # the EP, but explicitly prepare the same cast-folding graph.
                if not edge:
                    RemoveGraphAssertsPass()(ep.graph_module)
                ep = self._apply(
                    ep, fold_dtype_casts=True, allow_lossy_weight_casts=True
                )
                norm = self._assert_fused(ep, True, dtype)
                aten = exir_ops.edge.aten if edge else torch.ops.aten
                rounded = norm.args[2]
                self.assertEqual(rounded.target, aten._to_copy.default)
                self.assertEqual(rounded.kwargs["dtype"], dtype)
                gamma = rounded.args[0]
                self.assertEqual(gamma.target, aten.add.Tensor)
                self.assertEqual(gamma.meta["val"].dtype, torch.float32)
                source = gamma.args[0]
                if not edge:
                    self.assertEqual(source.target, torch.ops.aten.to.dtype)
                    self.assertEqual(source.args[1], torch.float32)
                    source = source.args[0]
                self.assertEqual(source.op, "placeholder")
                self.assertIn(source.name, ep.graph_signature.user_inputs)
                self.assertFalse(ep.constants)
                for current_weight in (weight, weight + 0.0001):
                    expected = (current_weight.float() + 1).to(dtype)
                    early_cast = (current_weight.to(dtype).float() + 1).to(dtype)
                    self.assertFalse(torch.equal(expected, early_cast))
                    torch.testing.assert_close(
                        ep.module()(x, current_weight),
                        torch.rms_norm(x, [16], expected, 1e-6),
                        rtol=0,
                        atol=0,
                    )

    def test_raw_metadata_assertions_preserve_cast_boundaries(self):
        """Treat raw metadata assertions as real users of FP32 intermediates."""
        model = _LossyWeightRMSNorm()
        x = _rmsnorm_input(torch.float16)
        weight = torch.linspace(-1.3, 1.7, 16)
        ep = _export_ep(model, (x, weight))
        assertions = _find_nodes(
            ep.graph_module, torch.ops.aten._assert_tensor_metadata.default
        )
        if not assertions:
            self.skipTest("This exporter does not emit tensor metadata assertions")
        ep = self._apply(ep, fold_dtype_casts=True, allow_lossy_weight_casts=True)
        norm = self._assert_fused(ep, True, torch.float32)
        self.assertTrue(any(norm in node.args for node in assertions))
        self.assertFalse(ep.constants)
        self._assert_close(ep.module()(x, weight), model(x, weight))

    def test_lossy_weight_casts_accept_fp32_scale(self):
        """Allow an existing FP32 scale as well as computed effective weights."""
        for dtype, edge in product((torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, edge=edge):
                model = _CastRMSNorm().eval()
                x = _rmsnorm_input(dtype)
                ep = self._apply(
                    _export_ep(model, (x,), edge),
                    fold_dtype_casts=True,
                    allow_lossy_weight_casts=True,
                )
                self._assert_fused(ep, True, dtype)
                self._assert_no_casts(ep)
                expected = torch.rms_norm(x, [16], model.weight.to(dtype), 1e-6)
                torch.testing.assert_close(ep.module()(x), expected, rtol=0, atol=0)

    def test_lossy_weight_casts_preserve_shared_boundaries(self):
        """Preserve escaping FP32 values and the original shared weight."""
        for escape in ("input", "gamma", "scaled", "weight"):
            with self.subTest(escape=escape):
                dtype = torch.bfloat16
                model = _OffsetRMSNorm(escape=escape).to(dtype).eval()
                original_weight = model.weight.detach().clone()
                x = _rmsnorm_input(dtype)
                ep = _to_edge_ep(model, (x,))
                casts = _find_nodes(
                    ep.graph_module, exir_ops.edge.aten._to_copy.default
                )
                ep = self._apply(
                    ep, fold_dtype_casts=True, allow_lossy_weight_casts=True
                )
                self._assert_fused(
                    ep, True, dtype if escape == "weight" else torch.float32
                )
                if escape != "weight":
                    self.assertEqual(
                        _find_nodes(
                            ep.graph_module, exir_ops.edge.aten._to_copy.default
                        ),
                        casts,
                    )
                actual, escaped = ep.module()(x)
                expected, expected_escaped = model(x)
                self._assert_close(actual, expected)
                torch.testing.assert_close(escaped, expected_escaped, rtol=0, atol=0)
                torch.testing.assert_close(
                    model.weight, original_weight, rtol=0, atol=0
                )

    def test_lossy_weight_casts_preserve_invalid_activation_boundaries(self):
        """Keep non-pure or mismatched activation casts even with opt-in."""
        for position in (0, 2):
            with self.subTest(position=position):
                model = _CastRMSNorm().to(torch.bfloat16).eval()
                ep = _export_ep(model, (_rmsnorm_input(torch.bfloat16),))
                casts = _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default)
                casts[position].kwargs = dict(casts[position].kwargs, non_blocking=True)
                ep = self._apply(
                    ep, fold_dtype_casts=True, allow_lossy_weight_casts=True
                )
                self._assert_fused(ep, with_scale=True)
                self.assertEqual(
                    _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default), casts
                )
        model = _CastRMSNorm(output_dtype=torch.bfloat16).to(torch.float16).eval()
        ep = _export_ep(model, (_rmsnorm_input(torch.float16),))
        casts = _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default)
        ep = self._apply(ep, fold_dtype_casts=True, allow_lossy_weight_casts=True)
        self._assert_fused(ep, with_scale=True)
        self.assertEqual(
            _find_nodes(ep.graph_module, torch.ops.aten._to_copy.default), casts
        )

    def test_lossy_weight_casts_dynamic_sequence(self):
        """Reuse a rounded-weight program across supported sequence lengths."""
        dtype = torch.bfloat16
        model = _OffsetRMSNorm().to(dtype).eval()
        ep = _export_ep(
            model,
            (_rmsnorm_input(dtype),),
            dynamic_shapes={"x": {1: torch.export.Dim("seq", min=1, max=8)}},
        )
        ep = self._apply(ep, fold_dtype_casts=True, allow_lossy_weight_casts=True)
        self._assert_fused(ep, True, dtype)
        for seq in (1, 4, 7):
            x = _rmsnorm_input(dtype, seq)
            expected = torch.rms_norm(
                x, [16], (model.weight.float() + 1.0).to(dtype), 1e-6
            )
            torch.testing.assert_close(ep.module()(x), expected, rtol=0, atol=0)

    def test_effective_weight_rounding_can_change_output(self):
        """Expose and bound the extra rounding against FP32 effective weights."""
        for dtype, delta in ((torch.float16, 2**-11), (torch.bfloat16, 2**-8)):
            with self.subTest(dtype=dtype):
                model = _OffsetRMSNorm().to(dtype).eval()
                with torch.no_grad():
                    model.weight.fill_(delta)
                x = _rmsnorm_input(dtype)
                gamma = model.weight.float() + 1.0
                self.assertTrue(torch.all(gamma != gamma.to(dtype).float()))
                ep = self._apply(
                    _export_ep(model, (x,)),
                    fold_dtype_casts=True,
                    allow_lossy_weight_casts=True,
                )
                actual, original = ep.module()(x), model(x)
                self.assertGreater((actual.float() - original.float()).abs().max(), 0)
                self._assert_close(actual, original)

    def test_noop_on_non_rms_norm(self):
        """Preserve unrelated programs under the default and lossy policies."""

        class M(nn.Module):
            def forward(self, x):
                return x + 1

        x = torch.randn(4, 4)
        for edge, fold, allow in (
            (False, False, False),
            (True, False, False),
            (False, True, True),
            (True, True, True),
        ):
            with self.subTest(edge=edge, fold=fold, allow=allow):
                ep = _export_ep(M(), (x,), edge)
                self._assert_noop(
                    ep,
                    FuseRMSNormPass(
                        fold_dtype_casts=fold, allow_lossy_weight_casts=allow
                    ),
                )
                torch.testing.assert_close(ep.module()(x), x + 1)

    def test_lossless_rewrite_updates_output_signature(self):
        """Refresh output names even when no derived weight is introduced."""
        # The default policy also rewrites graph outputs, without folding casts
        # or preparing constants. Both exposed outputs must remain executable.
        for edge in (False, True):
            with self.subTest(edge=edge, fp32_output=True):
                model = _FP32RMSNorm().eval()
                # Distinct output dtypes prevent export from aliasing both
                # outputs to an identity cast, keeping a direct FP32 output.
                x = _rmsnorm_input(torch.float16)
                ep = _export_ep(model, (x,), edge)
                original_output = ep.graph_signature.output_specs[-1].arg.name
                ep = self._apply(ep)
                norm = self._assert_fused(ep, True)
                self.assertNotEqual(norm.name, original_output)
                self.assertEqual(
                    ep.graph_signature.output_specs[-1].arg.name, norm.name
                )
                ep.validate()
                torch.testing.assert_close(
                    ep.module()(x), model(x), rtol=2e-6, atol=2e-6
                )

        for edge, with_scale, fold in product((False, True), repeat=3):
            with self.subTest(edge=edge, scale=with_scale, fold=fold):
                model = _CastRMSNorm(with_scale=with_scale).to(torch.float16).eval()
                x = _rmsnorm_input(torch.float16)
                ep = _export_ep(model, (x,), edge)
                original_inputs = str(ep.graph_signature.input_specs)
                original_output = ep.graph_signature.output_specs[0].arg.name
                ep = self._apply(ep, fold_dtype_casts=fold)
                self.assertEqual(str(ep.graph_signature.input_specs), original_inputs)
                if fold:
                    self.assertNotEqual(
                        ep.graph_signature.output_specs[0].arg.name, original_output
                    )
                    self.assertEqual(
                        ep.graph_signature.output_specs[0].arg.name,
                        _find_nodes(ep.graph_module, torch.ops.aten.rms_norm.default)[
                            0
                        ].name,
                    )
                ep.validate()
                self._assert_close(ep.module()(x), model(x))

    def test_mixed_exported_program_pass_manager(self):
        """Dispatch GraphModule and EP passes together while preserving signatures."""
        for edge, lossy in product((False, True), repeat=2):
            with self.subTest(edge=edge, lossy=lossy):
                model = _OffsetRMSNorm().to(torch.bfloat16).eval()
                x = _rmsnorm_input(torch.bfloat16)
                ep = _export_ep(model, (x,), edge)
                seen = []

                def before(gm, seen=seen):
                    self.assertIsInstance(gm, torch.fx.GraphModule)
                    self.assertFalse(_find_nodes(gm, torch.ops.aten.rms_norm.default))
                    seen.append("before")
                    return PassResult(gm, False)

                def after(gm, seen=seen):
                    self.assertIsInstance(gm, torch.fx.GraphModule)
                    self.assertEqual(
                        len(_find_nodes(gm, torch.ops.aten.rms_norm.default)), 1
                    )
                    seen.append("after")
                    return PassResult(gm, False)

                transform = FuseRMSNormPass(
                    fold_dtype_casts=True, allow_lossy_weight_casts=lossy
                )
                self.assertIsInstance(transform, ExportedProgramPassBase)
                manager = ExportedProgramPassManager([before, transform, after])
                result = manager(ep)
                self.assertTrue(result.modified)
                ep = result.exported_program
                self.assertEqual(seen, ["before", "after"])
                self._assert_signature(ep)
                if lossy:
                    _, weight = self._prepared_weight(ep)
                    torch.testing.assert_close(
                        ep.module()(x),
                        torch.rms_norm(x, [16], weight, 1e-6),
                        rtol=0,
                        atol=0,
                    )
                else:
                    self._assert_close(ep.module()(x), model(x))
                second = ExportedProgramPassManager([transform, after])(ep)
                self.assertFalse(second.modified)
                self._assert_signature(second.exported_program)


class TestFrozenRMSNormWeights(_RMSNormAssertions, unittest.TestCase):
    """Verify backend-independent frozen preparation and storage safety."""

    def _prepare(self, ep):
        return self._apply(ep, fold_dtype_casts=True, allow_lossy_weight_casts=True)

    def test_supported_frozen_sources_and_shared_storage(self):
        """Prepare supported frozen sources without changing shared original data."""
        for dtype, source, expression in product(
            (torch.float16, torch.bfloat16),
            ("parameter", "buffer", "constant"),
            ("direct", "add", "lifted_left", "lifted_right"),
        ):
            with self.subTest(dtype=dtype, source=source, expression=expression):
                weight_dtype = torch.float32 if expression == "direct" else dtype
                model = _LossyWeightRMSNorm(
                    weight_dtype,
                    source=source,
                    expression=expression,
                    shared=expression != "direct",
                )
                x = _rmsnorm_input(dtype)
                ep = _to_edge_ep(model, (x,))
                originals = {**ep.state_dict, **ep.constants}
                snapshots = {k: v.clone() for k, v in originals.items()}
                ep = self._prepare(ep)
                _, weight = self._prepared_weight(ep)
                expected = model.weight.float()
                if expression != "direct":
                    expected = expected + 1.0
                torch.testing.assert_close(weight, expected.to(dtype), rtol=0, atol=0)
                self.assertFalse(weight.requires_grad)
                current = {**ep.state_dict, **ep.constants}
                for key, value in originals.items():
                    self.assertIs(current[key], value)
                    torch.testing.assert_close(value, snapshots[key], rtol=0, atol=0)
                    self.assertNotEqual(value.data_ptr(), weight.data_ptr())
                self._assert_no_casts(ep)
                self.assertEqual(
                    _find_nodes(ep.graph_module, exir_ops.edge.aten.add.Tensor), []
                )
                actual = ep.module()(x)
                if model.shared:
                    self.assertTrue(
                        _find_nodes(ep.graph_module, exir_ops.edge.aten.sin.default)
                    )
                    actual, shared = actual
                    torch.testing.assert_close(
                        shared, model.weight.sin(), rtol=0, atol=0
                    )
                torch.testing.assert_close(
                    actual, torch.rms_norm(x, [16], expected.to(dtype), 1e-6)
                )
                self._assert_noop(
                    ep,
                    FuseRMSNormPass(
                        fold_dtype_casts=True, allow_lossy_weight_casts=True
                    ),
                )

    def test_runtime_weights_are_not_materialized(self):
        """Leave mutable, input-dependent, and unsupported expressions live."""
        for case in ("mutable", "input", "unknown", "alpha"):
            with self.subTest(case=case):
                model = _LossyWeightRMSNorm(
                    source="mutable" if case == "mutable" else "parameter",
                    expression=case if case in ("unknown", "alpha") else "add",
                )
                x = _rmsnorm_input(torch.bfloat16)
                inputs = (x, torch.randn(16)) if case == "input" else (x,)
                ep = _to_edge_ep(model, inputs)
                if case == "mutable":
                    self.assertTrue(ep.graph_signature.buffers_to_mutate)
                constants = tuple(ep.constants)
                ep = self._prepare(ep)
                self.assertEqual(tuple(ep.constants), constants)
                norm = _find_nodes(ep.graph_module, torch.ops.aten.rms_norm.default)[0]
                self.assertEqual(norm.args[2].op, "call_function")
                self.assertEqual(ep.module()(*inputs).shape, x.shape)

    def test_raw_inplace_weight_mutations_are_not_materialized(self):
        """Keep mutable raw-export weights live, including writes through a view."""

        class AliasedMutableNorm(_LossyWeightRMSNorm):
            def forward(self, x):
                result = super().forward(x)
                self.weight.view(-1).add_(0.125)
                return result

        for model in (
            _LossyWeightRMSNorm(source="mutable"),
            AliasedMutableNorm(source="buffer"),
        ):
            with self.subTest(model=type(model).__name__):
                x = _rmsnorm_input(torch.bfloat16)
                ep = export(model, (x,), strict=False)
                if not _find_nodes(ep.graph_module, torch.ops.aten.add_.Tensor):
                    self.skipTest("This exporter already functionalizes buffer writes")
                self.assertFalse(ep.graph_signature.buffers_to_mutate)
                RemoveGraphAssertsPass()(ep.graph_module)
                constants = tuple(ep.constants)
                ep = self._prepare(ep)
                self.assertEqual(tuple(ep.constants), constants)
                self.assertEqual(
                    len(_find_nodes(ep.graph_module, torch.ops.aten.add_.Tensor)), 1
                )
                module = ep.module()
                outputs = []
                for _ in range(2):
                    before = ep.state_dict["weight"].detach().clone()
                    expected = torch.rms_norm(
                        x, [16], (before.float() + 1).to(x.dtype), 1e-6
                    )
                    actual = module(x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        ep.state_dict["weight"], before + 0.125, rtol=0, atol=0
                    )
                    outputs.append(actual)
                self.assertFalse(torch.equal(outputs[0], outputs[1]))

    def test_collisions_dynamic_sequence_and_executable_signature(self):
        """Avoid existing names and retain executable dynamic input signatures."""
        model = _LossyWeightRMSNorm()
        x = _rmsnorm_input(torch.float16)
        ep = _to_edge_ep(model, (x,), {"x": {1: torch.export.Dim("seq", min=1, max=8)}})
        first_user = next(n for n in ep.graph.nodes if n.name == "x")
        with ep.graph.inserting_before(first_user):
            create_constant_placeholder(
                ep,
                ep.graph,
                "_rmsnorm_weight_0",
                InputKind.CONSTANT_TENSOR,
                torch.tensor([17.0]),
            )
        ep.state_dict["_rmsnorm_weight_1"] = torch.tensor([19.0])
        ep.constants["_rmsnorm_weight_2"] = torch.tensor([23.0])
        ep = self._prepare(ep)
        norm, weight = self._prepared_weight(ep)
        self.assertEqual(norm.args[2].name, "_rmsnorm_weight_3")
        self.assertEqual(ep.constants["_rmsnorm_weight_0"].item(), 17)
        self.assertEqual(ep.state_dict["_rmsnorm_weight_1"].item(), 19)
        self.assertEqual(ep.constants["_rmsnorm_weight_2"].item(), 23)
        for length in (1, 7):
            x = _rmsnorm_input(torch.float16, seq=length)
            torch.testing.assert_close(
                ep.module()(x), torch.rms_norm(x, [16], weight, 1e-6)
            )

    def test_prepared_weight_precedes_literal_user_inputs(self):
        """Insert derived constants before specialized scalar and tensor inputs."""

        class LiteralPrefixNorm(_OffsetRMSNorm):
            """Keep a specialized literal ahead of the activation input."""

            def forward(self, tag, x):
                """Normalize with a leading export-specialized argument."""
                return super().forward(x)

        # Exercise the raw EP: Edge's scalar lifting currently rejects a literal
        # prefix before RMSNorm fusion is reached.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = LiteralPrefixNorm().to(dtype).eval()
                x = _rmsnorm_input(dtype)
                ep = self._prepare(_export_ep(model, (7, x)))
                norm, weight = self._prepared_weight(ep)
                specs = ep.graph_signature.input_specs
                first_user = next(
                    index
                    for index, spec in enumerate(specs)
                    if spec.kind == InputKind.USER_INPUT
                )
                prepared_index = next(
                    index
                    for index, spec in enumerate(specs)
                    if spec.arg.name == norm.args[2].name
                )
                self.assertLess(prepared_index, first_user)
                self.assertEqual(specs[first_user].arg.name, "tag")
                ep.validate()
                expected_weight = (model.weight.float() + 1).to(dtype)
                torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)
                torch.testing.assert_close(
                    ep.module()(7, x),
                    torch.rms_norm(x, [16], expected_weight, 1e-6),
                    rtol=0,
                    atol=0,
                )
                self._assert_noop(
                    ep,
                    FuseRMSNormPass(
                        fold_dtype_casts=True, allow_lossy_weight_casts=True
                    ),
                )

    def test_fp32_add_precedes_rounding_without_fake_item(self):
        """Resolve real lifted ones and round only after the FP32 addition."""

        def evaluate_without_fake_item(*args):
            with patch.object(
                FakeTensor, "item", side_effect=AssertionError("FakeTensor.item")
            ):
                return _frozen_rmsnorm_weight(*args)

        for dtype, expression in product(
            (torch.float16, torch.bfloat16), ("add", "lifted_left", "lifted_right")
        ):
            with self.subTest(dtype=dtype, expression=expression):
                model = _LossyWeightRMSNorm(expression=expression)
                with torch.no_grad():
                    model.weight.copy_(torch.linspace(-1.0002, -0.9998, 16))
                x = _rmsnorm_input(dtype)
                ep = _to_edge_ep(model, (x,))
                # Scope the guard to frozen evaluation, not the epsilon matcher.
                with patch(
                    "executorch.backends.transforms.fuse_rms_norm._frozen_rmsnorm_weight",
                    side_effect=evaluate_without_fake_item,
                ):
                    ep = self._prepare(ep)
                _, weight = self._prepared_weight(ep)
                expected = (model.weight.float() + 1).to(dtype)
                early_cast = (model.weight.to(dtype).float() + 1).to(dtype)
                self.assertFalse(torch.equal(expected, early_cast))
                torch.testing.assert_close(weight, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    ep.module()(x),
                    torch.rms_norm(x, [16], expected, 1e-6),
                    rtol=0,
                    atol=0,
                )

    def test_frozen_upcast_with_different_activation_dtype(self):
        """Prepare upcast weights when the lossless folding path cannot apply."""
        for dtype, weight_dtype in (
            (torch.float16, torch.bfloat16),
            (torch.bfloat16, torch.float16),
        ):
            with self.subTest(dtype=dtype):
                model = _LossyWeightRMSNorm(weight_dtype, expression="direct")
                x = _rmsnorm_input(dtype)
                ep = self._prepare(_to_edge_ep(model, (x,)))
                _, weight = self._prepared_weight(ep)
                torch.testing.assert_close(
                    weight, model.weight.float().to(dtype), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    ep.module()(x), torch.rms_norm(x, [16], weight, 1e-6)
                )

    def test_preexisting_norm_weight_cast_is_not_materialized(self):
        """Do not evaluate casts belonging to preexisting low-precision norms."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(16))

            def forward(self, x):
                return torch.rms_norm(x, [16], self.weight.to(x.dtype), 1e-6)

        x = _rmsnorm_input(torch.float16)
        model = M()
        ep = export(model, (x,))
        self._assert_noop(
            ep, FuseRMSNormPass(fold_dtype_casts=True, allow_lossy_weight_casts=True)
        )
        self.assertFalse(ep.constants)
        torch.testing.assert_close(ep.module()(x), model(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
