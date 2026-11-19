# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.addmm_mm_to_linear import (
    addmm_can_be_linear,
    apply_addmm_mm_to_linear_transform,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim


def _dequantize(xq: torch.Tensor) -> torch.Tensor:
    return torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
        xq, torch.tensor(0.1), torch.tensor(0), -128, 127, torch.int8
    )


def _int8(*shape: int) -> torch.Tensor:
    return torch.randint(-128, 127, shape, dtype=torch.int8)


class _DequantizeLinearView(torch.nn.Module):
    def __init__(self, swap: bool) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
        self.swap = swap

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        a, b = xq.shape[0], xq.shape[1]
        y = self.linear(_dequantize(xq).view(-1, 16))
        return y.view(b, a, 8) if self.swap else y.view(a, b, 8)


class _DequantizeLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        return self.linear(_dequantize(xq).view(-1, 16))


class _DequantizeLinearNoOpView(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        return self.linear(_dequantize(xq).view(-1, 16)).view(-1, 8)


class _DequantizeLinearReadTwice(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, xq: torch.Tensor):
        y = self.linear(_dequantize(xq).view(-1, 16))
        return y, y.view(xq.shape[0], xq.shape[1], 8)


class _DequantizeSelfProduct(torch.nn.Module):
    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        flat = _dequantize(xq).view(-1, 16)
        return (flat @ flat.t()).view(xq.shape[0], xq.shape[1], -1)


class _DequantizeLinearBias(torch.nn.Module):
    def __init__(self, input_as_bias: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 16))
        self.bias = torch.nn.Parameter(torch.randn(6, 16))
        self.input_as_bias = input_as_bias

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        flat = _dequantize(xq).view(-1, 16)
        bias = flat if self.input_as_bias else self.bias
        return torch.nn.functional.linear(flat, self.weight, bias).view(
            xq.shape[0], xq.shape[1], 16
        )


class _Addmm(torch.nn.Module):
    def __init__(
        self, beta: float = 1.0, alpha: float = 1.0, bias_shape: tuple = (8,)
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 16))
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.beta = beta
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(
            self.bias, x, self.weight.t(), beta=self.beta, alpha=self.alpha
        )


class _StandaloneAddmm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 8))
        self.bias = torch.nn.Parameter(torch.randn(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x, self.weight)


class _PermutedWeightMatmul(torch.nn.Module):
    def __init__(self, dims: tuple, use_bias: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 16))
        self.bias = torch.nn.Parameter(torch.randn(16))
        self.dims = dims
        self.use_bias = use_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (
            self.weight.t() if self.dims is None else self.weight.permute(*self.dims)
        )
        return torch.addmm(self.bias, x, weight) if self.use_bias else x @ weight


class _DequantizeLinearViewAndFlatInput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, xq: torch.Tensor):
        flat = _dequantize(xq).view(-1, 16)
        y = self.linear(flat).view(xq.shape[0], xq.shape[1], 8)
        return y, flat * 2


class _DequantizeTwoLinearViews(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = torch.nn.Linear(16, 8)
        self.k = torch.nn.Linear(16, 8)

    def forward(self, xq: torch.Tensor):
        x = _dequantize(xq)
        a, b = xq.shape[0], xq.shape[1]
        return (
            self.q(x.view(-1, 16)).view(a, b, 8),
            self.k(x.view(-1, 16)).view(a, b, 8),
        )


_BOUNDED = ({0: Dim("a", min=2, max=8), 1: Dim("b", min=2, max=8)},)
_UNBOUNDED = ({0: Dim("a"), 1: Dim("b")},)


_DEQUANTIZE = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor
_ADDMM = exir_ops.edge.aten.addmm.default
_LINEAR = exir_ops.edge.aten.linear.default
_VIEW = exir_ops.edge.aten.view_copy.default


class TestAddmmMmToLinear(unittest.TestCase):
    def _transform(self, model, example, dynamic_shapes=None, compile_config=None):
        ep = torch.export.export(model, (example,), dynamic_shapes=dynamic_shapes)
        program = to_edge(ep, compile_config=compile_config).exported_program()
        apply_addmm_mm_to_linear_transform(program.graph_module.graph)
        return program

    def _linears(self, program):
        return [n for n in program.graph_module.graph.nodes if n.target == _LINEAR]

    def test_view_to_the_linear_output_shape_is_removed(self):
        model = _DequantizeLinearView(swap=False)
        for name, dynamic_shapes in (
            ("static", None),
            ("bounded", _BOUNDED),
            ("unbounded", _UNBOUNDED),
        ):
            with self.subTest(name):
                ep = torch.export.export(
                    model, (_int8(2, 3, 16),), dynamic_shapes=dynamic_shapes
                )
                program = to_edge(ep).exported_program()
                graph = program.graph_module.graph
                view_val = graph.output_node().args[0][0].meta["val"]
                apply_addmm_mm_to_linear_transform(graph)
                output = graph.output_node().args[0][0]
                self.assertEqual(output.target, _LINEAR)
                self.assertEqual(output.args[0].target, _DEQUANTIZE)
                self.assertIs(output.meta["val"], view_val)
                x = _int8(2, 3, 16)
                torch.testing.assert_close(program.module()(x), model(x))

    def test_view_to_another_shape_with_equal_bounds_is_kept(self):
        model = _DequantizeLinearView(swap=True)
        # [b, a, 8] and the linear output [a, b, 8] have the same upper bounds,
        # and without upper bounds the same example sizes.
        for name, example, dynamic_shapes in (
            ("bounded", _int8(2, 3, 16), _BOUNDED),
            ("unbounded", _int8(3, 3, 16), _UNBOUNDED),
        ):
            with self.subTest(name):
                program = self._transform(model, example, dynamic_shapes)
                output = program.graph_module.graph.output_node().args[0][0]
                self.assertEqual(output.target, _VIEW)
                (linear,) = self._linears(program)
                self.assertEqual(linear.args[0].target, _VIEW)
                x = _int8(2, 3, 16)
                torch.testing.assert_close(program.module()(x), model(x))

    def test_linear_reads_the_dequantized_input_only_when_its_view_goes(self):
        # The first linear has no view after it, and the second has another
        # reader besides its view, so both keep reading the flattened input. The
        # third has a view only to its own output shape, which goes, while the
        # linear keeps reading the flattened input. The fourth reads the
        # dequantized tensor once its view goes, while the other reader of the
        # flattened input keeps its shape.
        for model, reads in (
            (_DequantizeLinear(), _VIEW),
            (_DequantizeLinearReadTwice(), _VIEW),
            (_DequantizeLinearNoOpView(), _VIEW),
            (_DequantizeLinearViewAndFlatInput(), _DEQUANTIZE),
        ):
            for name, dynamic_shapes in (("static", None), ("bounded", _BOUNDED)):
                with self.subTest(type(model).__name__, shapes=name):
                    x = _int8(2, 3, 16)
                    program = self._transform(model, x, dynamic_shapes)
                    (linear,) = self._linears(program)
                    self.assertEqual(linear.args[0].target, reads)
                    self.assertIs(
                        program.graph_module.graph.output_node().args[0][0], linear
                    )
                    torch.testing.assert_close(program.module()(x), model(x))

    def test_weight_that_is_also_the_input_is_kept(self):
        # flat @ flat.t() becomes a linear whose input and weight are one node.
        # Only the input is pointed at the dequantized tensor when the view goes.
        model = _DequantizeSelfProduct()
        x = _int8(2, 3, 16)
        program = self._transform(model, x)
        (linear,) = self._linears(program)
        self.assertEqual(linear.args[0].target, _DEQUANTIZE)
        self.assertEqual(linear.args[1].target, _VIEW)
        self.assertIs(program.graph_module.graph.output_node().args[0][0], linear)
        # Products of 2 dequantized inputs reach about 1e3, and the 3-D linear
        # and the 2-D matmul round them differently.
        torch.testing.assert_close(program.module()(x), model(x), rtol=1e-4, atol=1e-3)

    def test_bias_with_more_than_one_dim_keeps_the_input(self):
        # A linear kept from export can add a bias per row of the flattened
        # input, which does not broadcast to the dequantized shape, so the
        # linear and its view stay.
        config = EdgeCompileConfig(preserve_ops=[torch.ops.aten.linear.default])
        for input_as_bias in (False, True):
            with self.subTest(input_as_bias=input_as_bias):
                model = _DequantizeLinearBias(input_as_bias)
                x = _int8(2, 3, 16)
                program = self._transform(model, x, compile_config=config)
                (linear,) = self._linears(program)
                self.assertEqual(linear.args[0].target, _VIEW)
                output = program.graph_module.graph.output_node().args[0][0]
                self.assertEqual(output.target, _VIEW)
                torch.testing.assert_close(program.module()(x), model(x))

    def test_only_an_addmm_without_scales_becomes_a_linear(self):
        # linear cannot scale its bias or its product, so only an addmm with
        # beta and alpha of 1 becomes a linear.
        for beta, alpha, linears in ((1.0, 1.0, 1), (2.0, 1.0, 0), (1.0, 0.5, 0)):
            with self.subTest(beta=beta, alpha=alpha):
                model = _Addmm(beta, alpha)
                x = torch.randn(6, 16)
                program = self._transform(model, x)
                self.assertEqual(len(self._linears(program)), linears)
                torch.testing.assert_close(program.module()(x), model(x))

    def test_only_an_addmm_of_a_transposed_weight_can_be_linear(self):
        # A standalone addmm(b, x, w) has no transpose for a linear to absorb.
        for model, expected in ((_Addmm(), True), (_StandaloneAddmm(), False)):
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model, (torch.randn(6, 16),))
                graph = to_edge(ep).exported_program().graph
                (addmm,) = [n for n in graph.nodes if n.target == _ADDMM]
                self.assertEqual(addmm_can_be_linear(addmm), expected)

    def test_only_an_addmm_with_a_1d_bias_becomes_a_linear(self):
        # ExecuTorch's linear kernel takes only a 1-D bias, while addmm takes
        # any bias that broadcasts to the output.
        for bias_shape, linears in (
            ((8,), 1),
            ((1,), 1),
            ((1, 8), 0),
            ((6, 8), 0),
            ((6, 1), 0),
            ((), 0),
        ):
            with self.subTest(bias_shape=bias_shape):
                model = _Addmm(bias_shape=bias_shape)
                x = torch.randn(6, 16)
                program = self._transform(model, x)
                self.assertEqual(len(self._linears(program)), linears)
                torch.testing.assert_close(program.module()(x), model(x))

    def test_only_a_transposed_weight_becomes_a_linear_weight(self):
        # permute(0, 1) leaves the weight as it is, so it is not a transpose.
        keep_t = EdgeCompileConfig(
            _check_ir_validity=False, preserve_ops=[torch.ops.aten.t.default]
        )
        for dims, config, linears in (
            ((0, 1), None, 0),
            ((1, 0), None, 1),
            ((-1, -2), None, 1),
            (None, keep_t, 1),
        ):
            for use_bias in (True, False):
                with self.subTest(dims=dims, use_bias=use_bias):
                    model = _PermutedWeightMatmul(dims, use_bias)
                    x = torch.randn(6, 16)
                    program = self._transform(model, x, compile_config=config)
                    self.assertEqual(len(self._linears(program)), linears)
                    torch.testing.assert_close(program.module()(x), model(x))

    def test_linears_sharing_a_dequantized_input_both_lose_their_views(self):
        model = _DequantizeTwoLinearViews()
        for name, dynamic_shapes in (("static", None), ("bounded", _BOUNDED)):
            with self.subTest(name):
                x = _int8(2, 3, 16)
                program = self._transform(model, x, dynamic_shapes)
                linears = self._linears(program)
                self.assertEqual(len(linears), 2)
                self.assertTrue(all(n.args[0].target == _DEQUANTIZE for n in linears))
                self.assertFalse(
                    any(n.target == _VIEW for n in program.graph_module.graph.nodes)
                )
                torch.testing.assert_close(program.module()(x), model(x))


if __name__ == "__main__":
    unittest.main()
