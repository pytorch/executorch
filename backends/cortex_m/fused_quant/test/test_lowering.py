# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Cortex-M fused_quant recipe lowers every CMSIS-NN op.

Each case exports a model twice with the same quantizer: through the Cortex-M
recipe, and through the fused_quant recipe, which stops at the fused edge
program. Both programs run on their ops' reference kernels, so comparing them
checks the lowering's fixed-point parameters against the fused ops they replace.
"""

import dataclasses
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from executorch.backends.cortex_m.fused_quant.quantizer import (
    get_cortex_m_fused_quant_quantizer,
)
from executorch.backends.cortex_m.fused_quant.recipe import (
    get_cortex_m_fused_quant_recipe,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT16_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
)
from executorch.backends.transforms.fused_quant.passes import get_fused_quant_passes
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_manager import PassType
from executorch.export import (
    export,
    ExportRecipe,
    ExportSession,
    LoweringRecipe,
    StageType,
)
from executorch.frontends.fused_quant.quantizer.quantizer import FusedQuantQuantizer
from executorch.frontends.fused_quant.recipe import (
    get_fused_quant_edge_compile_config,
    get_fused_quant_quantization_recipe,
)
from parameterized import parameterized  # type: ignore[import-untyped]
from torch import fx, nn
from torch._prims_common import make_contiguous_strides_for
from torch.export import ExportedProgram
from torch.nn import functional as F
from torchao.quantization.pt2e.quantizer import QuantizationSpec

_aten = exir_ops.edge.aten
_cortex_m = exir_ops.edge.cortex_m


class _Unary(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class _Binary(nn.Module):
    def __init__(
        self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fn(x, y)


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.linear = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv(x)))
        return torch.softmax(self.linear(x.flatten(1)), -1)


class _SamePadding(nn.Module):
    """A stride-2 convolution padded to SAME by hand, as a converted
    TensorFlow model pads it."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Conv2d(4, 8, 3, padding=1)
        self.second = nn.Conv2d(8, 8, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(F.pad(torch.relu(self.first(x)), (0, 1, 0, 1)))


class _Attention(nn.Module):
    def __init__(self, is_causal: bool = False) -> None:
        super().__init__()
        self.is_causal = is_causal

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query, key, value, is_causal=self.is_causal
        )


class _SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x, x, x, need_weights=False)[0]


class _Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Conv2d(4, 8, 3, padding=1)
        self.second = nn.Conv2d(4, 8, 1)
        self.linear = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.first(x) + self.second(x))
        return self.linear(y.mean((2, 3)))


class _NHWCRowConv(nn.Module):
    """A 1xN convolution, padded by hand, of NHWC data one row high."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 8, (1, 5), stride=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x.permute(0, 3, 1, 2), (1, 2)))


@dataclass(frozen=True)
class _Case:
    name: str
    model: Callable[[], nn.Module]
    shapes: tuple[tuple[int, ...], ...]
    # The cortex_m ops the lowered program must contain.
    expected: tuple[str, ...]
    # The inputs drawn from [0.5, 1.5), for divisors and square roots.
    positive: tuple[int, ...] = ()
    per_channel_linear: bool = False
    activation: QuantizationSpec = INT8_ACTIVATION_PER_TENSOR_QSPEC
    # How far the lowered program may stray from the fused one, in output
    # steps: a fixed-point kernel can round the other way than the float
    # reference; a table or a comparison cannot. A folded activation drops the
    # rounding to the grid between the two ops.
    tolerance: int = 1
    # The cortex_m ops the lowered program must not contain.
    absent: tuple[str, ...] = ()


_IMAGE = (1, 4, 8, 8)

_CASES = [
    _Case(
        "add",
        lambda: _Binary(torch.add),
        (_IMAGE, _IMAGE),
        ("quantized_add", "quantize_per_tensor", "dequantize_per_tensor"),
    ),
    _Case("sub", lambda: _Binary(torch.sub), (_IMAGE, _IMAGE), ("quantized_add",)),
    _Case(
        "add_channel_broadcast",
        lambda: _Binary(torch.add),
        (_IMAGE, (1, 4, 1, 1)),
        ("quantized_add",),
    ),
    _Case(
        "add_relu",
        lambda: _Binary(lambda x, y: torch.relu(x + y)),
        (_IMAGE, _IMAGE),
        ("quantized_add",),
        tolerance=3,
        absent=("quantized_activation",),
    ),
    _Case("mul", lambda: _Binary(torch.mul), (_IMAGE, _IMAGE), ("quantized_mul",)),
    _Case(
        "mul_innermost_broadcast",
        lambda: _Binary(torch.mul),
        ((2, 3, 16), (16,)),
        ("quantized_mul",),
    ),
    _Case(
        "div",
        lambda: _Binary(torch.div),
        (_IMAGE, _IMAGE),
        ("quantized_div",),
        positive=(1,),
    ),
    _Case(
        "div_int16",
        lambda: _Binary(torch.div),
        (_IMAGE, _IMAGE),
        ("quantized_div",),
        positive=(1,),
        activation=INT16_ACTIVATION_PER_TENSOR_QSPEC,
    ),
    _Case(
        "minimum",
        lambda: _Binary(torch.minimum),
        (_IMAGE, _IMAGE),
        ("minimum",),
        tolerance=0,
    ),
    _Case(
        "maximum",
        lambda: _Binary(torch.maximum),
        (_IMAGE, _IMAGE),
        ("maximum",),
        tolerance=0,
    ),
    _Case(
        "sigmoid",
        lambda: _Unary(torch.sigmoid),
        (_IMAGE,),
        ("quantized_activation",),
        tolerance=0,
    ),
    _Case(
        "hardswish",
        lambda: _Unary(F.hardswish),
        (_IMAGE,),
        ("quantized_activation",),
        tolerance=0,
    ),
    _Case(
        "rsqrt",
        lambda: _Unary(torch.rsqrt),
        (_IMAGE,),
        ("quantized_activation",),
        positive=(0,),
        tolerance=0,
    ),
    _Case(
        "pow",
        lambda: _Unary(lambda x: torch.pow(x, 2)),
        (_IMAGE,),
        ("quantized_activation",),
        tolerance=0,
    ),
    _Case(
        "clamp_to_constant_tensors",
        lambda: _Unary(
            lambda x: torch.clamp(x, torch.full_like(x, -1.0), torch.full_like(x, 1.0))
        ),
        (_IMAGE,),
        ("quantized_activation",),
        tolerance=0,
    ),
    _Case(
        "softmax",
        lambda: _Unary(lambda x: torch.softmax(x, -1)),
        ((2, 4, 16),),
        ("softmax",),
    ),
    _Case(
        "softmax_over_channels",
        lambda: _Unary(lambda x: torch.softmax(x, 1)),
        ((2, 3, 4),),
        ("softmax", "transpose"),
    ),
    _Case(
        "pad",
        lambda: _Unary(lambda x: F.pad(x, (1, 2, 0, 1))),
        (_IMAGE,),
        ("pad",),
        tolerance=0,
    ),
    _Case("linear", lambda: nn.Linear(16, 8), ((2, 16),), ("quantized_linear",)),
    _Case(
        "linear_relu6",
        lambda: nn.Sequential(nn.Linear(16, 8), nn.ReLU6()),
        ((2, 16),),
        ("quantized_linear",),
        tolerance=3,
        absent=("quantized_activation",),
    ),
    _Case(
        "linear_hardsigmoid",
        lambda: nn.Sequential(nn.Linear(16, 8), nn.Hardsigmoid()),
        ((2, 16),),
        ("quantized_linear",),
        tolerance=3,
        absent=("quantized_activation",),
    ),
    _Case(
        "linear_per_channel",
        lambda: nn.Linear(16, 8),
        ((2, 16),),
        ("quantized_linear_per_channel",),
        per_channel_linear=True,
    ),
    _Case(
        "bmm",
        lambda: _Binary(torch.bmm),
        ((2, 4, 8), (2, 8, 6)),
        ("quantized_batch_matmul", "transpose"),
    ),
    _Case(
        "conv2d",
        lambda: nn.Conv2d(4, 8, 3, padding=1),
        (_IMAGE,),
        ("quantized_conv2d_nhwc", "transpose"),
    ),
    _Case(
        "conv2d_relu",
        lambda: nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU()),
        (_IMAGE,),
        ("quantized_conv2d_nhwc",),
        tolerance=3,
        absent=("quantized_activation",),
    ),
    _Case(
        "conv2d_same_padding",
        _SamePadding,
        (_IMAGE,),
        ("quantized_conv2d_nhwc",),
        tolerance=3,
        absent=("pad", "quantized_activation"),
    ),
    _Case(
        "conv1d",
        lambda: nn.Conv1d(4, 8, 3, padding=1),
        ((1, 4, 16),),
        ("quantized_conv2d_nhwc",),
    ),
    _Case(
        "depthwise_conv2d",
        lambda: nn.Conv2d(4, 8, 3, padding=1, groups=4),
        (_IMAGE,),
        ("quantized_depthwise_conv2d_nhwc",),
    ),
    _Case(
        "transpose_conv2d",
        lambda: nn.ConvTranspose2d(4, 8, 3, stride=2),
        (_IMAGE,),
        ("quantized_transpose_conv2d_nhwc",),
    ),
    _Case(
        "avg_pool2d",
        lambda: nn.AvgPool2d(2),
        (_IMAGE,),
        ("quantized_avg_pool2d_nhwc",),
    ),
    _Case(
        "avg_pool2d_counting_padding",
        lambda: nn.AvgPool2d(3, stride=1, padding=1),
        (_IMAGE,),
        ("pad", "quantized_avg_pool2d_nhwc"),
    ),
    _Case(
        "spatial_mean",
        lambda: _Unary(lambda x: x.mean((2, 3), keepdim=True)),
        (_IMAGE,),
        ("transpose", "quantized_avg_pool2d_nhwc"),
    ),
    _Case(
        "max_pool2d",
        lambda: nn.MaxPool2d(2),
        (_IMAGE,),
        ("quantized_max_pool2d_nhwc",),
        tolerance=0,
    ),
    _Case(
        "max_pool2d_relu",
        lambda: nn.Sequential(nn.MaxPool2d(2), nn.ReLU()),
        (_IMAGE,),
        ("quantized_max_pool2d_nhwc",),
        tolerance=0,
        absent=("quantized_activation",),
    ),
]

# These run the CMSIS-NN kernel of their _nhwc twin on a channels_last
# dim order. The fused_quant layout is explicit, so the recipe emits the twins.
_LAYOUT_TWINS = {
    "quantized_conv2d": "quantized_conv2d_nhwc",
    "quantized_depthwise_conv2d": "quantized_depthwise_conv2d_nhwc",
    "quantized_transpose_conv2d": "quantized_transpose_conv2d_nhwc",
    "quantized_avg_pool2d": "quantized_avg_pool2d_nhwc",
    "quantized_max_pool2d": "quantized_max_pool2d_nhwc",
}


def _inputs(
    shapes: tuple[tuple[int, ...], ...], positive: tuple[int, ...] = ()
) -> list[tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    return [
        tuple(
            torch.rand(shape) + 0.5 if index in positive else torch.randn(shape)
            for index, shape in enumerate(shapes)
        )
        for _ in range(4)
    ]


def _edge_program(session: ExportSession) -> ExportedProgram:
    return session.get_edge_program_manager().exported_program()


def _lower(
    model: nn.Module,
    inputs: list[tuple[torch.Tensor, ...]],
    per_channel_linear: bool = False,
    activation: QuantizationSpec = INT8_ACTIVATION_PER_TENSOR_QSPEC,
) -> ExportedProgram:
    """The lowered edge program, before ``to_executorch`` swaps in out variants."""
    recipe = dataclasses.replace(
        get_cortex_m_fused_quant_recipe(
            per_channel_linear=per_channel_linear, activation=activation
        ),
        pipeline_stages=[
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
        ],
    )
    return _edge_program(
        export(model=model, example_inputs=inputs, export_recipe=recipe)
    )


def _targets(program: ExportedProgram) -> set[object]:
    return {node.target for node in program.graph.nodes if node.op == "call_function"}


def _output_scale(program: ExportedProgram) -> float:
    """The scale of the dequantize behind the program's single output."""
    (node,) = cast(tuple[fx.Node], program.graph.output_node().args[0])
    while node.target != _cortex_m.dequantize_per_tensor.default:
        node = cast(fx.Node, node.args[0])
    return float(cast(float, node.args[1]))


def _fused_recipe(quantizer: FusedQuantQuantizer) -> ExportRecipe:
    """The Cortex-M recipe without the cortex_m passes: the fused_quant edge
    program the lowering starts from, as a reference."""

    def edge_passes(method_name: str, program: ExportedProgram) -> list[PassType]:
        return get_fused_quant_passes(channels_last=True)

    return ExportRecipe(
        name="cortex_m_fused_reference",
        strict=False,
        quantization_recipe=get_fused_quant_quantization_recipe(quantizer),
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


class LoweringTest(unittest.TestCase):
    def _assert_matches_fused_program(
        self,
        model: nn.Module,
        inputs: list[tuple[torch.Tensor, ...]],
        lowered: ExportedProgram,
        tolerance: int,
        per_channel_linear: bool = False,
        activation: QuantizationSpec = INT8_ACTIVATION_PER_TENSOR_QSPEC,
    ) -> None:
        fused = _edge_program(
            export(
                model=model,
                example_inputs=inputs,
                export_recipe=_fused_recipe(
                    get_cortex_m_fused_quant_quantizer(
                        per_channel_linear=per_channel_linear, activation=activation
                    )
                ),
            )
        )
        example = inputs[0]
        torch.testing.assert_close(
            lowered.module()(*example),
            fused.module()(*example),
            atol=tolerance * _output_scale(lowered) * 1.001,
            rtol=0,
        )

    @parameterized.expand([(case.name, case) for case in _CASES])  # type: ignore[misc]
    def test_lowers_to_cmsis_nn(self, _name: str, case: _Case) -> None:
        torch.manual_seed(0)
        model = case.model().eval()
        inputs = _inputs(case.shapes, case.positive)

        lowered = _lower(model, inputs, case.per_channel_linear, case.activation)

        targets = _targets(lowered)
        for name in case.expected:
            self.assertIn(getattr(_cortex_m, name).default, targets)
        for name in case.absent:
            self.assertNotIn(getattr(_cortex_m, name).default, targets)
        self._assert_matches_fused_program(
            model,
            inputs,
            lowered,
            case.tolerance,
            case.per_channel_linear,
            case.activation,
        )

    def test_an_op_the_kernels_cannot_run_falls_back_to_portable(self) -> None:
        """quantized_add broadcasts only an operand of one value per channel,
        so this add decomposes instead."""
        model = _Binary(torch.add)
        inputs = _inputs((_IMAGE, (1, 1, 8, 8)))

        lowered = _lower(model, inputs)

        targets = _targets(lowered)
        self.assertNotIn(_cortex_m.quantized_add.default, targets)
        self.assertIn(_aten.add.Tensor, targets)
        self._assert_matches_fused_program(model, inputs, lowered, tolerance=0)

    def test_attention_runs_on_the_kernels(self) -> None:
        """An unmasked attention decomposes to bmm -> softmax -> bmm, with its
        scale folded into the first bmm."""
        torch.manual_seed(0)
        model = _Attention().eval()
        inputs = _inputs(((1, 2, 8, 16),) * 3)

        lowered = _lower(model, inputs)

        targets = _targets(lowered)
        self.assertIn(_cortex_m.quantized_batch_matmul.default, targets)
        self.assertIn(_cortex_m.softmax.default, targets)
        self.assertNotIn(_aten.where.self, targets)
        self.assertNotIn(_cortex_m.quantized_activation.default, targets)
        self._assert_matches_fused_program(model, inputs, lowered, tolerance=2)

    def test_multihead_attention_runs_on_the_kernels(self) -> None:
        """nn.MultiheadAttention views its attention's output, which the
        decomposition lays out differently."""
        torch.manual_seed(0)
        model = _SelfAttention().eval()
        inputs = _inputs(((1, 8, 32),))

        lowered = _lower(model, inputs)

        targets = _targets(lowered)
        self.assertIn(_cortex_m.quantized_batch_matmul.default, targets)
        self.assertIn(_cortex_m.softmax.default, targets)
        self._assert_matches_fused_program(model, inputs, lowered, tolerance=3)

    def test_residual_add_runs_in_nhwc(self) -> None:
        """The add of two convolutions, its relu folded in, reads and writes
        NHWC, and the pool and the linear after it read its grid, so only the
        input is permuted and nothing requantizes."""
        torch.manual_seed(0)
        model = _Residual().eval()
        inputs = _inputs(((1, 4, 8, 8),))

        lowered = _lower(model, inputs)

        targets = [
            node.target for node in lowered.graph.nodes if node.op == "call_function"
        ]
        self.assertIn(_cortex_m.quantized_add.default, targets)
        self.assertIn(_cortex_m.quantized_avg_pool2d_nhwc.default, targets)
        self.assertNotIn(_cortex_m.quantized_activation.default, targets)
        self.assertEqual(targets.count(_cortex_m.transpose.default), 1)
        self._assert_matches_fused_program(model, inputs, lowered, tolerance=4)

    def test_a_row_of_nhwc_data_stays_contiguous(self) -> None:
        """The pad kernel reads a contiguous tensor of one channel or one pixel
        as NHWC, as an NHWC row is, so the row pads as 3-d: every tensor keeps
        the contiguous layout the convolution kernel requires."""
        torch.manual_seed(0)
        model = _NHWCRowConv().eval()
        inputs = _inputs(((1, 1, 8, 4),))

        lowered = _lower(model, inputs)

        targets = _targets(lowered)
        self.assertIn(_cortex_m.pad.default, targets)
        self.assertIn(_cortex_m.quantized_conv2d_nhwc.default, targets)
        for node in lowered.graph.nodes:
            val = node.meta.get("val")
            if node.op == "call_function" and isinstance(val, torch.Tensor):
                self.assertEqual(
                    tuple(val.stride()),
                    tuple(make_contiguous_strides_for(val.shape)),
                    node.name,
                )
        self._assert_matches_fused_program(model, inputs, lowered, tolerance=1)

    def test_masked_attention_runs_in_float(self) -> None:
        """A causal mask's -inf cannot be quantized, so the attention stays
        float, as in the current flow."""
        torch.manual_seed(0)
        model = _Attention(is_causal=True).eval()
        inputs = _inputs(((1, 2, 8, 16),) * 3)

        lowered = _lower(model, inputs)

        targets = _targets(lowered)
        self.assertNotIn(_cortex_m.softmax.default, targets)
        self.assertIn(_aten._softmax.default, targets)
        example = inputs[0]
        with torch.no_grad():
            expected = model(*example)
        torch.testing.assert_close(
            lowered.module()(*example),
            expected,
            atol=0.05 * expected.abs().max().item(),
            rtol=0,
        )

    def test_global_average_pool_reads_the_convolution_output(self) -> None:
        """AdaptiveAvgPool2d(1) traces to a spatial mean, which pools the
        convolution's NHWC output rather than permuting it back. The linear
        reads the pool's grid through the flatten, so nothing requantizes."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        ).eval()
        inputs = _inputs(((1, 3, 8, 8),))

        lowered = _lower(model, inputs)

        targets = [
            node.target for node in lowered.graph.nodes if node.op == "call_function"
        ]
        self.assertIn(_cortex_m.quantized_avg_pool2d_nhwc.default, targets)
        self.assertNotIn(_cortex_m.quantized_activation.default, targets)
        # Only the convolution's input is permuted.
        self.assertEqual(targets.count(_cortex_m.transpose.default), 1)
        # One output step each for the convolution, the pool and the linear
        # can meet at the output; the per-op case bounds the mean alone.
        self._assert_matches_fused_program(model, inputs, lowered, tolerance=2)

    def test_model_matches_the_float_model(self) -> None:
        torch.manual_seed(0)
        model = _ConvNet().eval()
        inputs = _inputs(((1, 3, 8, 8),))

        lowered = _lower(model, inputs)

        targets = _targets(lowered)
        for name in ("quantized_conv2d_nhwc", "quantized_linear", "softmax"):
            self.assertIn(getattr(_cortex_m, name).default, targets)
        (example,) = inputs[0]
        with torch.no_grad():
            expected = model(example)
        torch.testing.assert_close(
            lowered.module()(example),
            expected,
            atol=0.1 * expected.abs().max().item(),
            rtol=0,
        )

    def test_model_exports_to_a_program(self) -> None:
        """The recipe runs through to_executorch, onto the cortex_m out variants."""
        torch.manual_seed(0)
        session = export(
            model=_ConvNet().eval(),
            example_inputs=_inputs(((1, 3, 8, 8),)),
            export_recipe=get_cortex_m_fused_quant_recipe(),
        )

        self.assertTrue(session.get_pte_buffer())
        targets = _targets(session.get_executorch_program_manager().exported_program())
        self.assertIn(torch.ops.cortex_m.quantized_conv2d_nhwc.out, targets)


class CoverageTest(unittest.TestCase):
    def test_every_cortex_m_op_has_a_case(self) -> None:
        registered = {
            schema.name.removeprefix("cortex_m::")
            for schema in torch._C._jit_get_all_schemas()
            if schema.name.startswith("cortex_m::")
        }
        covered = {name for case in _CASES for name in case.expected}
        covered |= {op for op, twin in _LAYOUT_TWINS.items() if twin in covered}

        self.assertEqual(registered - covered, set())
