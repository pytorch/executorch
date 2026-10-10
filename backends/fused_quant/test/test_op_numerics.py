# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from collections.abc import Callable
from dataclasses import dataclass, field

import executorch.backends.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.fused_quant.decompose_fused_quant import DecomposeFusedQuant
from executorch.backends.fused_quant.graph_utils import split_fused_arg_names
from executorch.backends.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.fused_quant.test.helpers import create_per_tensor_qparams
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_manager import ExportedProgramPassManager
from parameterized import parameterized
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree

_QPARAM_FIELDS = ("scale", "zero_point", "dtype", "quant_min", "quant_max")
_INP_SCALE = 0.25
_OUT_SCALE = 0.5


def _int8(*shape: int) -> Callable[[], torch.Tensor]:
    return lambda: torch.randint(-128, 128, shape, dtype=torch.int8)


def _float(*shape: int) -> Callable[[], torch.Tensor]:
    return lambda: torch.randn(*shape)


@dataclass(frozen=True)
class _Case:
    op: str
    operands: dict[str, Callable[[], torch.Tensor]]
    quantized: frozenset[str]
    extras: dict[str, object] = field(default_factory=dict)
    float_out: bool = True

    def overload(self) -> str:
        return self.op.partition(".")[2] or "default"

    def packet_name(self) -> str:
        return self.op.partition(".")[0]


_CONV_2D = {
    "stride": [1, 1],
    "padding": [0, 0],
    "dilation": [1, 1],
    "transposed": False,
    "output_padding": [0, 0],
    "groups": 1,
}
_POOL_2D = {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]}

_CASES: list[_Case] = [
    _Case("requantize", {"inp": _int8(1, 8)}, frozenset({"inp"}), float_out=False),
    *(
        _Case(name, {"inp": _int8(1, 8)}, frozenset({"inp"}))
        for name in (
            "relu",
            "hardswish",
            "sigmoid",
            "tanh",
            "hard_tanh",
            "silu",
            "hardsigmoid",
        )
    ),
    _Case("gelu", {"inp": _int8(1, 8)}, frozenset({"inp"}), {"approximate": "none"}),
    *(
        _Case(
            name,
            {"inp": _int8(2, 4), "other": _int8(2, 4)},
            frozenset({"inp", "other"}),
        )
        for name in ("add", "mul", "sub")
    ),
    _Case("add.Scalar", {"inp": _int8(2, 4)}, frozenset({"inp"}), {"other": 2.0}),
    _Case("mul.Scalar", {"inp": _int8(2, 4)}, frozenset({"inp"}), {"other": 2.0}),
    _Case("sub.Scalar", {"inp": _int8(2, 4)}, frozenset({"inp"}), {"other": 2.0}),
    _Case(
        "bmm",
        {"inp": _int8(2, 3, 4), "other": _int8(2, 4, 5)},
        frozenset({"inp", "other"}),
    ),
    _Case(
        "linear",
        {"inp": _int8(2, 8), "weight": _int8(4, 8), "bias": _float(4)},
        frozenset({"inp", "weight"}),
    ),
    _Case(
        "convolution",
        {"inp": _int8(1, 2, 6, 6), "weight": _int8(3, 2, 3, 3), "bias": _float(3)},
        frozenset({"inp", "weight"}),
        _CONV_2D,
    ),
    _Case(
        "convolution_channels_last",
        {"inp": _int8(1, 6, 6, 2), "weight": _int8(3, 3, 3, 2), "bias": _float(3)},
        frozenset({"inp", "weight"}),
        _CONV_2D,
    ),
    _Case(
        "conv1d",
        {"inp": _int8(1, 2, 8), "weight": _int8(3, 2, 3), "bias": _float(3)},
        frozenset({"inp", "weight"}),
        {"stride": [1], "padding": [0], "dilation": [1], "groups": 1},
    ),
    _Case(
        "conv2d",
        {"inp": _int8(1, 2, 6, 6), "weight": _int8(3, 2, 3, 3), "bias": _float(3)},
        frozenset({"inp", "weight"}),
        {"stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1},
    ),
    _Case(
        "conv3d",
        {
            "inp": _int8(1, 2, 4, 4, 4),
            "weight": _int8(3, 2, 3, 3, 3),
            "bias": _float(3),
        },
        frozenset({"inp", "weight"}),
        {"stride": [1, 1, 1], "padding": [0, 0, 0], "dilation": [1, 1, 1], "groups": 1},
    ),
    *(
        _Case(
            name,
            {"inp": _int8(*shape)},
            frozenset({"inp"}),
            {**_POOL_2D, "dilation": [1, 1], "ceil_mode": False},
        )
        for name, shape in (
            ("max_pool2d_with_indices", (1, 2, 6, 6)),
            ("max_pool2d_with_indices_channels_last", (1, 6, 6, 2)),
        )
    ),
    *(
        _Case(
            name,
            {"inp": _int8(*shape)},
            frozenset({"inp"}),
            {
                **_POOL_2D,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None,
            },
        )
        for name, shape in (
            ("avg_pool2d", (1, 2, 6, 6)),
            ("avg_pool2d_channels_last", (1, 6, 6, 2)),
        )
    ),
    _Case(
        "native_layer_norm",
        {"inp": _int8(2, 8), "weight": _float(8), "bias": _float(8)},
        frozenset({"inp"}),
        {"normalized_shape": [8], "eps": 1e-5},
    ),
    _Case(
        "rms_norm",
        {"inp": _int8(2, 8), "weight": _float(8)},
        frozenset({"inp"}),
        {"normalized_shape": [8], "eps": 1e-5},
    ),
    _Case(
        "_masked_softmax",
        {"inp": _int8(2, 4), "mask": lambda: torch.rand(2, 4) > 0.5},
        frozenset({"inp"}),
        {"dim": -1, "mask_type": 2},
    ),
    _Case(
        "embedding",
        {"weight": _int8(10, 4), "indices": lambda: torch.tensor([0, 3, 9])},
        frozenset({"weight"}),
    ),
]

_VARIANTS: list[tuple[str, _Case, bool]] = [
    (
        f"{case.op}_{'float_out' if not quantize_out else 'quantized_out'}",
        case,
        quantize_out,
    )
    for case in _CASES
    for quantize_out in (True, False)
    if quantize_out or case.float_out
]


def _eager_op(case: _Case) -> torch._ops.OpOverload:
    return getattr(getattr(torch.ops.fused_quant, case.packet_name()), case.overload())


def _edge_op(case: _Case) -> object:
    return getattr(
        getattr(exir_ops.edge.fused_quant, case.packet_name()), case.overload()
    )


def _qparam_values(
    prefix: str, quantized: bool
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.dtype, int, int]:
    if not quantized:
        return (None, None, torch.float32, 0, 0)
    is_out = prefix == "out"
    return (
        torch.tensor(_OUT_SCALE if is_out else _INP_SCALE),
        torch.tensor(0, dtype=torch.int64),
        torch.int8 if is_out else torch.float32,
        -128,
        127,
    )


def _build_args(
    case: _Case,
    quantize_out: bool,
    operands: dict[str, object],
    qparams: Callable[[str, bool], tuple[object, ...]],
) -> list[object]:
    """Lay out ``case``'s arguments in schema order.

    Tensor operands come from ``operands``, each qparam block from ``qparams``,
    and every remaining argument from ``case.extras`` or its schema default.
    """
    op = _eager_op(case)
    input_names, _ = split_fused_arg_names(op)
    blocks = {
        prefix: qparams(
            prefix, quantize_out if prefix == "out" else prefix in case.quantized
        )
        for prefix in (*input_names, "out")
    }
    extras = dict(case.extras)
    args: list[object] = []
    for argument in op._schema.arguments:
        name = argument.name
        if name in operands:
            args.append(operands[name])
            continue
        block_field = next(
            (
                (block, f)
                for block in blocks
                for f in _QPARAM_FIELDS
                if name == f"{block}_{f}"
            ),
            None,
        )
        if block_field is not None:
            block, f = block_field
            args.append(blocks[block][_QPARAM_FIELDS.index(f)])
        elif name in extras:
            args.append(extras.pop(name))
        else:
            assert argument.has_default_value(), f"{case.op}: no value for {name}"
            args.append(argument.default_value)
    assert not extras, f"{case.op}: unused extras {sorted(extras)}"
    return args


def _first_output(result: object) -> torch.Tensor:
    return result[0] if isinstance(result, (tuple, list)) else result


class FakeKernelDtypeTest(unittest.TestCase):
    @parameterized.expand(_VARIANTS)
    def test_fake_matches_eager(
        self, _name: str, case: _Case, quantize_out: bool
    ) -> None:
        torch.manual_seed(0)
        operands = {name: make() for name, make in case.operands.items()}
        args = _build_args(case, quantize_out, operands, _qparam_values)
        eager = _eager_op(case)(*args)

        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            fake_args = pytree.tree_map_only(torch.Tensor, mode.from_tensor, args)
            fake = _eager_op(case)(*fake_args)

        eager_flat, _ = pytree.tree_flatten(eager)
        fake_flat, _ = pytree.tree_flatten(fake)
        self.assertEqual(len(eager_flat), len(fake_flat))
        for e, f in zip(eager_flat, fake_flat):
            self.assertEqual(f.dtype, e.dtype)
            self.assertEqual(f.shape, e.shape)


class DecomposeNumericsTest(unittest.TestCase):
    """The reference kernel and its DecomposeFusedQuant lowering agree exactly."""

    @parameterized.expand(_VARIANTS)
    def test_decomposed_matches_reference(
        self, _name: str, case: _Case, quantize_out: bool
    ) -> None:
        torch.manual_seed(0)
        inputs = {name: make() for name, make in case.operands.items()}
        builder = ProgramBuilder()
        nodes = {
            name: builder.placeholder(name, value) for name, value in inputs.items()
        }

        def lifted_qparams(prefix: str, quantized: bool) -> tuple[object, ...]:
            if not quantized:
                return _qparam_values(prefix, quantized)
            scale = _OUT_SCALE if prefix == "out" else _INP_SCALE
            dtype = torch.int8 if prefix == "out" else torch.float32
            return create_per_tensor_qparams(builder, scale=scale, dtype=dtype)

        args = _build_args(case, quantize_out, nodes, lifted_qparams)
        result = builder.call_operator(_edge_op(case), tuple(args))
        if isinstance(result.data, (tuple, list)):
            result = builder.call_getitem(result, 0)
        builder.output([result])
        program = builder.get_program()

        user_inputs = [inputs[name] for name in case.operands]
        reference = program.module()(*user_inputs)
        # The documented fallback: channels-last ops go back to channels-first
        # before decomposition, which only knows the ATen layouts.
        decomposed = ExportedProgramPassManager(
            [ToChannelsFirst(), DecomposeFusedQuant()]
        )(program).exported_program

        self.assertFalse(
            any(
                getattr(node.target, "namespace", None) == "fused_quant"
                for node in decomposed.graph.nodes
                if node.op == "call_function"
            )
        )
        torch.testing.assert_close(
            _first_output(decomposed.module()(*user_inputs)),
            _first_output(reference),
            rtol=0,
            atol=0,
        )
