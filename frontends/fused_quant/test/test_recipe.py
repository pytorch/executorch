# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from dataclasses import replace

import torch
from executorch.backends.transforms.fused_quant.decompose_fused_quant import (
    AssertNoFusedQuantOps,
    DecomposeFusedQuant,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.transforms.fused_quant.passes import get_fused_quant_passes
from executorch.exir import ExportedProgram
from executorch.exir._warnings import ExperimentalWarning
from executorch.exir.pass_manager import PassType
from executorch.export import (
    export,
    ExportRecipe,
    ExportSession,
    LoweringRecipe,
    StageType,
)
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    get_core_aten_quantizers,
    OpQuantizer,
    QuantizerBase,
)
from executorch.frontends.fused_quant.recipe import (
    get_fused_quant_aten_transform_passes,
    get_fused_quant_edge_compile_config,
    get_fused_quant_portable_recipe,
    get_fused_quant_quantization_recipe,
)
from parameterized import parameterized
from torch import nn
from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer import QuantizationConfig, QuantizationSpec


class ConvReluLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(8 * 4 * 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        return self.linear(x.flatten(1))


class CallCounter(nn.Module):
    """Counts its calls in a buffer, the way a KV cache tracks its position."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("calls", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.add_(1)
        return self.linear(x)


class SoftmaxAttention(nn.Module):
    """SDPA, which traces to aten._safe_softmax."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class ConvLinearSlices(nn.Module):
    """A convolution, then a linear with a scalar add whose output is sliced, as a
    fused QKV projection's is: something for every pipeline flag to rewrite."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.linear = nn.Linear(8 * 4 * 4, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(torch.relu(self.conv(x)).flatten(1)) + 1.0
        q, k, v = y.split(4, dim=-1)
        return q * k + v


def _calibration_inputs() -> list[tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    return [(torch.randn(2, 3, 4, 4),) for _ in range(4)]


def _qconfig(weight_per_channel: bool) -> QuantizationConfig:
    activation = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
    )
    weight = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=(
            torch.per_channel_symmetric
            if weight_per_channel
            else torch.per_tensor_symmetric
        ),
        ch_axis=0 if weight_per_channel else None,
        observer_or_fake_quant_ctr=(
            PerChannelMinMaxObserver.with_args(eps=2**-12)
            if weight_per_channel
            else MinMaxObserver
        ),
    )
    return QuantizationConfig(activation, activation, weight, None)


def _quantizer(*extra: QuantizerBase) -> FusedQuantQuantizer:
    return FusedQuantQuantizer(
        [
            *extra,
            *get_core_aten_quantizers(
                _qconfig(weight_per_channel=False),
                _qconfig(weight_per_channel=True),
            ),
        ]
    )


def _linear_quantizer() -> OpQuantizer:
    """Keeps the non-core `aten.linear` whole and fuses it."""
    return OpQuantizer(
        torch.ops.aten.linear.default,
        torch.ops.fused_quant.linear.default,
        _qconfig(weight_per_channel=True),
        activation_names=("input",),
        weight_names=("weight",),
        other_names=("bias",),
    )


def _export(
    model: nn.Module,
    inputs: list[tuple[torch.Tensor, ...]],
    quantizer: FusedQuantQuantizer | None = None,
) -> ExportSession:
    return export(
        model=model,
        example_inputs=inputs,
        export_recipe=get_fused_quant_portable_recipe(quantizer or _quantizer()),
    )


def _backend_recipe(quantizer: FusedQuantQuantizer) -> ExportRecipe:
    """A backend's recipe built from the fused_quant pieces, stopped at the fused
    edge program its colorer and lowering would take."""

    def edge_passes(method_name: str, program: ExportedProgram) -> list[PassType]:
        return get_fused_quant_passes()

    return ExportRecipe(
        name="backend",
        strict=False,
        quantization_recipe=get_fused_quant_quantization_recipe(quantizer),
        aten_transform_passes=get_fused_quant_aten_transform_passes(quantizer),
        lowering_recipe=LoweringRecipe(
            edge_transform_passes=[edge_passes],
            edge_compile_config=get_fused_quant_edge_compile_config(
                quantizer.preserved_ops()
            ),
        ),
        pipeline_stages=[
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
        ],
    )


def _flagged_recipe(quantizer: FusedQuantQuantizer, **flags: bool) -> ExportRecipe:
    """The portable recipe with the pipeline's opt-in rewrites turned on."""

    def edge_passes(method_name: str, program: ExportedProgram) -> list[PassType]:
        return [
            *get_fused_quant_passes(**flags),
            ToChannelsFirst(),
            DecomposeFusedQuant(),
            AssertNoFusedQuantOps(),
        ]

    return ExportRecipe(
        name="flagged",
        strict=False,
        quantization_recipe=get_fused_quant_quantization_recipe(quantizer),
        aten_transform_passes=get_fused_quant_aten_transform_passes(quantizer),
        lowering_recipe=LoweringRecipe(
            edge_transform_passes=[edge_passes],
            edge_compile_config=get_fused_quant_edge_compile_config(()),
        ),
    )


_FLAGS = (
    "channels_last",
    "fold_into_linear",
    "prequantize_embedding",
    "split_linear_at_slices",
    "absorb_quant_with_fork",
    "absorb_quant_into_inputs",
)


def _targets(graph: torch.fx.Graph) -> list[str]:
    return [str(node.target) for node in graph.nodes if node.op == "call_function"]


class PortableRecipeTest(unittest.TestCase):
    def test_quantizes_into_fused_quant_ops(self) -> None:
        session = _export(ConvReluLinear(), _calibration_inputs())

        program = session.get_stage_artifacts()[StageType.TORCH_EXPORT].data["forward"]
        targets = _targets(program.graph)
        self.assertTrue(
            any("fused_quant" in target for target in targets),
            f"expected fused_quant ops, got {targets}",
        )

    def test_default_pipeline_converges(self) -> None:
        with self.assertNoLogs(
            "executorch.backends.transforms.fused_quant.pass_base", level="WARNING"
        ):
            _export(ConvReluLinear(), _calibration_inputs())

    def test_decomposes_every_fused_quant_op_without_delegating(self) -> None:
        session = _export(ConvReluLinear(), _calibration_inputs())

        targets = _targets(
            session.get_executorch_program_manager().exported_program().graph
        )
        self.assertFalse(
            any("fused_quant" in target for target in targets),
            f"fused_quant ops left over: {targets}",
        )
        self.assertNotIn("executorch_call_delegate", targets)

    def test_program_matches_the_float_model(self) -> None:
        """The .pte output tracks the eager float model within quantization error."""
        model = ConvReluLinear()
        inputs = _calibration_inputs()

        session = _export(model, inputs)

        (example,) = inputs[0]
        with torch.no_grad():
            expected = model(example)
        (actual,) = session.run_method("forward", (example,))
        self.assertEqual(actual.shape, expected.shape)
        # int8 per-tensor activations over this range; compare against the
        # magnitude of the reference rather than an absolute epsilon.
        tolerance = 0.1 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)

    def test_calibration_leaves_buffers_at_their_initial_values(self) -> None:
        model = CallCounter()

        session = _export(model, [(torch.randn(2, 4),) for _ in range(4)])

        self.assertEqual(model.get_buffer("calls").item(), 0)
        program = session.get_stage_artifacts()[StageType.TORCH_EXPORT].data["forward"]
        self.assertEqual(program.state_dict["calls"].item(), 0)

    def test_decomposes_a_non_core_op_the_quantizer_keeps_whole(self) -> None:
        model = ConvReluLinear()
        inputs = _calibration_inputs()

        session = _export(model, inputs, _quantizer(_linear_quantizer()))

        program = session.get_stage_artifacts()[StageType.TORCH_EXPORT].data["forward"]
        self.assertIn("fused_quant.linear.default", _targets(program.graph))
        self.assertNotIn(
            "aten.linear.default",
            _targets(session.get_executorch_program_manager().exported_program().graph),
        )
        (example,) = inputs[0]
        with torch.no_grad():
            expected = model(example)
        (actual,) = session.run_method("forward", (example,))
        tolerance = 0.1 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)


class BackendRecipeTest(unittest.TestCase):
    def test_keeps_the_fused_ops_for_the_backend(self) -> None:
        """A backend lowers the fused ops, so the pieces leave them in place."""
        session = export(
            model=ConvReluLinear(),
            example_inputs=_calibration_inputs(),
            export_recipe=_backend_recipe(_quantizer()),
        )

        targets = _targets(session.get_edge_program_manager().exported_program().graph)
        self.assertTrue(
            any("fused_quant" in target for target in targets),
            f"expected fused_quant ops, got {targets}",
        )
        self.assertNotIn(StageType.TO_EXECUTORCH, session.get_stage_artifacts())


class PipelineTest(unittest.TestCase):
    def _edge_targets(self, recipe: ExportRecipe) -> list[str]:
        session = export(
            model=ConvReluLinear(),
            example_inputs=_calibration_inputs(),
            export_recipe=recipe,
        )
        return _targets(session.get_edge_program_manager().exported_program().graph)

    def test_default_pipeline_rewrites_the_fused_graph(self) -> None:
        quantizer = _quantizer(_linear_quantizer())
        unoptimized = replace(
            _backend_recipe(quantizer),
            lowering_recipe=LoweringRecipe(
                edge_compile_config=get_fused_quant_edge_compile_config(
                    quantizer.preserved_ops()
                ),
            ),
        )

        self.assertNotEqual(
            self._edge_targets(_backend_recipe(quantizer)),
            self._edge_targets(unoptimized),
        )

    @parameterized.expand(
        [(flag, {flag: True}) for flag in _FLAGS]
        + [("all", dict.fromkeys(_FLAGS, True))]
    )
    def test_program_with_flags_matches_the_float_model(
        self, _name: str, flags: dict[str, bool]
    ) -> None:
        model = ConvLinearSlices()
        inputs = _calibration_inputs()

        with self.assertNoLogs(
            "executorch.backends.transforms.fused_quant.pass_base", level="WARNING"
        ):
            session = export(
                model=model,
                example_inputs=inputs,
                export_recipe=_flagged_recipe(_quantizer(_linear_quantizer()), **flags),
            )

        (example,) = inputs[0]
        with torch.no_grad():
            expected = model(example)
        (actual,) = session.run_method("forward", (example,))
        tolerance = 0.1 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)


class ExperimentalTest(unittest.TestCase):
    def test_portable_recipe_warns_that_it_is_experimental(self) -> None:
        with self.assertWarnsRegex(
            ExperimentalWarning, "fused_quant recipes are experimental"
        ):
            get_fused_quant_portable_recipe(_quantizer())


class _KeepsSafeSoftmax(QuantizerBase):
    """Quantizes nothing, but keeps `_safe_softmax` whole, as a backend that
    lowers or rewrites it would."""

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        return model

    def fuse(self, exported_program: ExportedProgram) -> bool:
        return False

    def preserved_ops(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.aten._safe_softmax.default]


class SafeSoftmaxTest(unittest.TestCase):
    """SDPA traces to `aten._safe_softmax`, which is not core ATen."""

    def _inputs(self) -> list[tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)
        shape = (1, 2, 8, 16)
        return [
            (torch.randn(shape), torch.randn(shape), torch.randn(shape))
            for _ in range(2)
        ]

    def test_kept_whole_for_a_quantizer_that_reports_it(self) -> None:
        """Kept whole, it reaches the edge graph, which accepts it."""
        session = export(
            model=SoftmaxAttention(),
            example_inputs=self._inputs(),
            export_recipe=_backend_recipe(_quantizer(_KeepsSafeSoftmax())),
        )

        edge_ops = _targets(session.get_edge_program_manager().exported_program().graph)
        self.assertTrue(
            any("_safe_softmax" in op for op in edge_ops),
            f"_safe_softmax did not reach the edge graph: {sorted(edge_ops)}",
        )

    def test_decomposed_for_the_core_aten_quantizers(self) -> None:
        """Nothing recomposes it, so the portable recipe decomposes it rather than
        leaving it for the edge verifier."""
        session = _export(SoftmaxAttention(), self._inputs())

        edge_ops = _targets(
            session.get_stage_artifacts()[StageType.TO_EDGE_TRANSFORM_AND_LOWER]
            .data.exported_program()
            .graph
        )
        self.assertFalse(
            any("_safe_softmax" in op for op in edge_ops),
            f"_safe_softmax reached the edge graph: {sorted(edge_ops)}",
        )


class CoreAtenQuantizersTest(unittest.TestCase):
    def test_every_quantized_op_is_core_aten(self) -> None:
        """Core ATen ops survive decomposition, so the capture never has to
        preserve them."""
        ops = _quantizer().preserved_ops()

        self.assertTrue(ops)
        for op in ops:
            self.assertIn(torch.Tag.core, op.tags, str(op))
