# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import executorch.exir.dialects.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
    get_zero_point,
)
from executorch.backends.transforms.fused_quant.optimization_passes.fold_relu_into_producer import (
    FoldReluIntoProducer,
)
from executorch.backends.transforms.fused_quant.test.helpers import (
    create_per_tensor_qparams,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

_LINEAR = exir_ops.edge.fused_quant.linear.default
_RELU = exir_ops.edge.fused_quant.relu.default


def _linear_relu(
    *,
    relu_inp: tuple[float, int] = (0.5, 10),
    relu_out: tuple[float, int] = (0.25, -128),
    expose_linear: bool = False,
) -> tuple[ExportedProgram, torch.Tensor]:
    """``linear (out 0.5/10) -> view_copy -> relu (relu_inp -> relu_out)``."""
    builder = ProgramBuilder()
    inp = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
    inp_node = builder.placeholder("inp", inp)
    weight = builder.placeholder(
        "weight",
        torch.tensor([[1, -2, 3, -4, 5, -6, 7, -8]] * 3, dtype=torch.int8)
        * torch.tensor([[1], [-1], [2]], dtype=torch.int8),
        input_kind=InputKind.CONSTANT_TENSOR,
    )
    linear = builder.call_operator(
        _LINEAR,
        (
            inp_node,
            weight,
            None,
            *create_per_tensor_qparams(builder, scale=0.25, dtype=torch.float32),
            *create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32),
            None,
            None,
            torch.float32,
            0,
            0,
            *create_per_tensor_qparams(builder, scale=0.5, zero_point=10),
        ),
    )
    view = builder.call_operator(
        exir_ops.edge.aten.view_copy.default, (linear, [1, 2, 3])
    )
    relu = builder.call_operator(
        _RELU,
        (
            view,
            *create_per_tensor_qparams(
                builder,
                scale=relu_inp[0],
                zero_point=relu_inp[1],
                dtype=torch.float32,
            ),
            *create_per_tensor_qparams(
                builder, scale=relu_out[0], zero_point=relu_out[1]
            ),
        ),
    )
    builder.output([relu, linear] if expose_linear else [relu])
    return builder.get_program(), inp


def _qparams(
    ep: ExportedProgram, node: torch.fx.Node, prefix: str
) -> tuple[float, int]:
    qparams = get_qparams_from_node(node, prefix)
    assert qparams is not None
    return get_scale(ep, qparams), get_zero_point(ep, qparams)


def _dequantized(output: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (output.to(torch.float32) - zero_point) * scale


class FoldReluIntoProducerTest(unittest.TestCase):
    def test_relu_clamping_at_quant_min_is_removed(self) -> None:
        ep, inp = _linear_relu()
        (expected,) = ep.module()(inp)

        result = FoldReluIntoProducer()(ep)

        self.assertTrue(result.modified)
        ep = result.exported_program
        self.assertEqual([], ep.graph.find_nodes(op="call_function", target=_RELU))
        (linear,) = ep.graph.find_nodes(op="call_function", target=_LINEAR)
        self.assertEqual((0.25, -128), _qparams(ep, linear, "out"))
        (actual,) = ep.module()(inp)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        # Only the rounding to the intermediate 0.5 scale is gone: at most 1 LSB.
        diff = _dequantized(actual, 0.25, -128) - _dequantized(expected, 0.25, -128)
        self.assertLessEqual(float(diff.abs().max()), 0.25)
        self.assertTrue(bool((actual >= -128).all()))
        ep.validate()

    def test_relu_with_a_real_clamp_is_kept_at_shared_qparams(self) -> None:
        ep, inp = _linear_relu(relu_out=(0.25, -100))
        (expected,) = ep.module()(inp)

        result = FoldReluIntoProducer()(ep)

        self.assertTrue(result.modified)
        ep = result.exported_program
        (relu,) = ep.graph.find_nodes(op="call_function", target=_RELU)
        (linear,) = ep.graph.find_nodes(op="call_function", target=_LINEAR)
        self.assertEqual((0.25, -100), _qparams(ep, linear, "out"))
        self.assertEqual((0.25, -100), _qparams(ep, relu, "inp"))
        self.assertEqual((0.25, -100), _qparams(ep, relu, "out"))
        (actual,) = ep.module()(inp)
        self.assertTrue(bool((actual >= -100).all()))
        diff = _dequantized(actual, 0.25, -100) - _dequantized(expected, 0.25, -100)
        self.assertLessEqual(float(diff.abs().max()), 0.25)

    def test_relu_already_sharing_its_producers_qparams_is_left_alone(self) -> None:
        ep, _ = _linear_relu(relu_out=(0.5, 10))

        self.assertFalse(FoldReluIntoProducer()(ep).modified)

    def test_relu_reading_at_other_qparams_than_written_is_left_alone(self) -> None:
        """Moving the producer would change what the relu reads."""
        ep, _ = _linear_relu(relu_inp=(0.4, 10))

        self.assertFalse(FoldReluIntoProducer()(ep).modified)

    def test_producer_with_another_reader_is_left_alone(self) -> None:
        ep, _ = _linear_relu(expose_linear=True)

        self.assertFalse(FoldReluIntoProducer()(ep).modified)
