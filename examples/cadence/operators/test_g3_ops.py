# pyre-unsafe
import unittest
from typing import Any, cast, List, OrderedDict, Tuple

from executorch.backends.cadence.utils import facto_util

from parameterized import parameterized

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import torch
import torch.nn as nn
from executorch.backends.cadence.aot.export_example import export_and_run_model


class ATenOpTestCases(unittest.TestCase):
    def run_and_verify(self, model: nn.Module, inputs: Tuple[Any, ...]) -> None:
        model.eval()
        export_and_run_model(
            model, inputs, file_name=self._testMethodName, run_and_compare=False
        )

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("add.Tensor")])
    @torch.no_grad()
    def test_g3_add_tensor_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class AddTensor(nn.Module):
            def __init__(self, alpha: float):
                super().__init__()
                self.alpha = alpha

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.add(x, y, alpha=self.alpha)

        model = AddTensor(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("add.Scalar")])
    @torch.no_grad()
    def test_aten_add_Scalar_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class AddScalar(nn.Module):
            def __init__(self, alpha: float):
                super().__init__()
                self.alpha = alpha

            def forward(self, x: torch.Tensor, y: float):
                return torch.add(x, y, alpha=self.alpha)

        inputs = posargs[:-1]  # posargs = [x_tensor, y_scalar, alpha_scalar]
        alpha = posargs[-1]
        model = AddScalar(alpha)

        self.run_and_verify(model, tuple(inputs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("sub.Tensor")])
    @torch.no_grad()
    def test_g3_sub_tensor_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class SubTensor(nn.Module):
            def __init__(self, alpha: float):
                super().__init__()
                self.alpha = alpha

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.sub(x, y, alpha=self.alpha)

        model = SubTensor(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("sub.Scalar")])
    @torch.no_grad()
    def test_g3_sub_scalar_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        # Tensor-Scalar subtraction
        class SubScalar(torch.nn.Module):
            def __init__(self, other):
                super().__init__()
                self.other = other

            def forward(self, x):
                return torch.ops.aten.sub.Scalar(x, self.other)

        inputs = posargs[0]  # posargs = [x_tensor, y_scalar, alpha_scalar]
        model = SubScalar(posargs[1])

        self.run_and_verify(model, (inputs,))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("div.Tensor")])
    @torch.no_grad()
    def test_g3_div_tensor_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class DivTensor(nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.div(x, y + 1)

        model = DivTensor(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("div.Scalar")])
    @torch.no_grad()
    def test_g3_div_scalar_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class DivScalar(nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.div(x, y + 1)

        model = DivScalar(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("exp.default")])
    @torch.no_grad()
    def test_g3_exp_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class Exp(nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.exp(x)

        model = Exp(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("mul.Tensor")])
    @torch.no_grad()
    def test_g3_mul_tensor_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class MulTensor(nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return x * y

        model = MulTensor(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("mul.Scalar")])
    @torch.no_grad()
    def test_g3_mul_scalar_out(
        self,
        posargs: List[str],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class MulScalar(nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return x * y

        model = MulScalar(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("native_layer_norm.default")])
    @torch.no_grad()
    def test_g3_native_layer_norm_out(
        self,
        posargs: List[int],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        inputs, normalized_shape, weight, bias, _ = posargs
        model = nn.LayerNorm(normalized_shape, eps=1e-5)
        if weight is not None:
            weight = cast(torch.Tensor, weight)
            model.weight = nn.Parameter(torch.rand_like(weight))
        if bias is not None:
            bias = cast(torch.Tensor, bias)
            model.bias = nn.Parameter(torch.rand_like(bias))

        self.run_and_verify(model, (inputs,))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("neg.default")])
    @torch.no_grad()
    def test_g3_neg_out(
        self,
        posargs: List[int],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class Neg(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.neg(x)

        model = Neg(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("rsqrt.default")])
    @torch.no_grad()
    def test_g3_rsqrt_out(
        self,
        posargs: List[int],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class Rsqrt(nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.ops.aten.rsqrt(x)

        model = Rsqrt(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("sigmoid.default")])
    @torch.no_grad()
    def test_g3_sigmoid_out(
        self,
        posargs: List[int],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        model = nn.Sigmoid(**inkwargs)

        self.run_and_verify(model, tuple(posargs))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("_softmax.default")])
    @torch.no_grad()
    def test_g3__softmax_out(
        self,
        posargs: List[int],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        inputs, _, _ = posargs
        model = nn.Softmax(dim=-1)

        self.run_and_verify(model, (inputs,))

    # pyre-ignore[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand([*facto_util.facto_testcase_gen("mean.dim")])
    def test_g3_mean_dim_out(
        self,
        posargs: List[int],
        inkwargs: OrderedDict[str, str],
    ) -> None:
        class Meandim(nn.Module):
            def forward(
                self,
                x: torch.Tensor,
                dim_list: Tuple[int],
                keepdim: bool,
                dtype: torch.dtype = torch.float32,
            ) -> torch.Tensor:
                return torch.ops.aten.mean.dim(
                    x,
                    dim_list,
                    keepdim,
                    dtype=dtype,
                )

        model = Meandim()

        self.run_and_verify(
            model,
            inputs=tuple(posargs),
        )


if __name__ == "__main__":
    unittest.main()
