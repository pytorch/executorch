# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from executorch.backends.xnnpack.partition.config.generic_node_configs import MulConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.test.tester import Tester, ToEdgeTransformAndLower
from torch.nn.utils.parametrizations import weight_norm


class TestFoldComputedParams(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class WeightNormConv1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = weight_norm(
                torch.nn.Conv1d(8, 8, 5, padding=2, groups=2), dim=2
            )

        def forward(self, x):
            return self.conv(x)

    class ComputedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            return F.linear(x, self.weight * 2, self.bias + 1)

    class FoldedNameCollision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))
            # The name the folded weight would get by default.
            self.aten_mul_tensor_folded = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            return F.linear(x, self.weight * 2, self.aten_mul_tensor_folded)

    class MatmulWeightConv1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 3))

        def forward(self, x):
            return F.conv1d(x, (self.weight @ self.weight).reshape(3, 3, 1))

    class NestedGemmWeightConv1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.randn(3, 3))
            self.b = torch.nn.Parameter(torch.randn(3, 3))

        def forward(self, x):
            return F.conv1d(x, (self.a @ self.b.flip([1])).reshape(3, 3, 1))

    class ScaledConv1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4, 3))

        def forward(self, x):
            return F.conv1d(x, self.weight * 2)

    class ScaledAddmm(torch.nn.Module):
        def __init__(self, computed):
            super().__init__()
            self.computed = computed
            self.weight = torch.nn.Parameter(torch.randn(8, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            weight, bias = self.weight, self.bias
            if self.computed:
                weight, bias = weight * 2, bias + 1
            return torch.addmm(bias, x, weight, alpha=2, beta=3)

    class SourceUsedElsewhere(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))

        def forward(self, x):
            return F.linear(x, self.weight * 2), self.weight.sum()

    class SourceRetainedByGemm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16))

        def forward(self, x):
            return F.linear(x, (self.weight * 2).unsqueeze(1), self.weight)

    class WeightComputationFeedsActivation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 16))

        def forward(self, x):
            w = self.weight * 2
            return w @ (w * 2) + x

    class LowRankWeight(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.randn(16, 2))
            self.b = torch.nn.Parameter(torch.randn(2, 16))

        def forward(self, x):
            return F.linear(x, self.a @ self.b)

    class MutatedBufferSource(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(4, 4, 3))

        def forward(self, x):
            y = F.conv1d(x, self.weight * 2)
            self.weight.copy_(x.mean() * torch.ones(4, 4, 3))
            return y

    def _test_fully_delegated(self, module, inputs):
        (
            Tester(module.eval(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_weight_norm_conv1d(self):
        self._test_fully_delegated(self.WeightNormConv1d(), (torch.randn(1, 8, 16),))

    def test_fp32_computed_linear_weight_and_bias(self):
        self._test_fully_delegated(self.ComputedLinear(), (torch.randn(2, 8),))

    def test_fp32_folded_name_does_not_replace_param(self):
        self._test_fully_delegated(self.FoldedNameCollision(), (torch.randn(2, 8),))

    def test_fp32_computed_weight_of_rejected_gemm_not_folded(self):
        # Conv1d with two dynamic dims is rejected after its weight was checked,
        # while the matmul computing the weight is still delegated on its own.
        dims = {
            0: torch.export.Dim("batch", min=1, max=4),
            2: torch.export.Dim("length", min=4, max=16),
        }
        for module in (self.MatmulWeightConv1d(), self.NestedGemmWeightConv1d()):
            with self.subTest(module=type(module).__name__):
                (
                    Tester(
                        module.eval(), (torch.randn(2, 3, 8),), dynamic_shapes=(dims,)
                    )
                    .export()
                    .to_edge_transform_and_lower()
                    .check(
                        ["executorch_exir_dialects_edge__ops_aten_convolution_default"]
                    )
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs()
                )

    def test_fp32_scaled_addmm_not_delegated(self):
        for computed in (False, True):
            with self.subTest(computed=computed):
                (
                    Tester(self.ScaledAddmm(computed).eval(), (torch.randn(2, 8),))
                    .export()
                    .to_edge_transform_and_lower()
                    .check(["executorch_exir_dialects_edge__ops_aten_addmm_default"])
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs()
                )

    def test_fp32_weight_computed_by_prior_delegate_not_folded(self):
        (
            Tester(self.ScaledConv1d().eval(), (torch.randn(1, 4, 8),))
            .export()
            .to_edge_transform_and_lower(
                ToEdgeTransformAndLower(
                    [XnnpackPartitioner(configs=[MulConfig]), XnnpackPartitioner()]
                )
            )
            .check(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_computed_weight_not_folded(self):
        for module, inputs, gemm in (
            (self.SourceUsedElsewhere(), (torch.randn(2, 8),), "aten_mm_default"),
            (self.LowRankWeight(), (torch.randn(2, 16),), "aten_mm_default"),
            (self.SourceRetainedByGemm(), (torch.randn(2, 1),), "aten_addmm_default"),
            (
                self.WeightComputationFeedsActivation(),
                (torch.randn(16, 16),),
                "aten_mm_default",
            ),
            (
                self.MutatedBufferSource(),
                (torch.randn(1, 4, 8),),
                "aten_convolution_default",
            ),
        ):
            with self.subTest(module=type(module).__name__):
                (
                    Tester(module.eval(), inputs)
                    .export()
                    .to_edge_transform_and_lower()
                    .check([f"executorch_exir_dialects_edge__ops_{gemm}"])
                )

    def test_fp32_externally_stored_weight_not_folded(self):
        tester = Tester(self.WeightNormConv1d().eval(), (torch.randn(1, 8, 16),))
        for node in tester.export().get_artifact().graph.nodes:
            if node.op == "placeholder" and node.name.startswith("p_"):
                node.meta["custom"] = {"delegate_constant_tag": "weights.ptd"}
        tester.to_edge_transform_and_lower().check(
            ["executorch_exir_dialects_edge__ops_aten_convolution_default"]
        )
