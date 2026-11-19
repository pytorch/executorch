# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from executorch.backends.xnnpack._passes.fold_constant_subgraphs import (
    FoldConstantSubgraphsPass,
)
from executorch.backends.xnnpack.partition.config.gemm_configs import AddmmConfig
from executorch.backends.xnnpack.partition.config.generic_node_configs import MulConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.test.tester import (
    Quantize,
    Tester,
    ToEdgeTransformAndLower,
)
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.nn.utils.parametrizations import weight_norm

# A custom op whose kernel is only available at runtime.
torch.library.define("xnnpack_test::fake_only", "(Tensor x) -> Tensor")


@torch.library.register_fake("xnnpack_test::fake_only")
def _(x):
    return torch.empty_like(x)


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

    class ComputedBiasLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 4)

        def forward(self, x):
            return F.linear(x, self.linear.weight, self.linear.bias * 2)

    class InputScaledBiasAddmm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 4))
            self.offset = torch.nn.Parameter(torch.randn(8, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.addmm(self.bias * x.mean(), x, self.weight * 2 + self.offset)

    class CustomOpWeight(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))

        def forward(self, x):
            return F.linear(x, torch.ops.xnnpack_test.fake_only(self.weight))

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
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.addmm(self.bias + 1, x, self.weight * 2, alpha=2, beta=3)

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

    def _test_fully_delegated(self, module, inputs, folded=()):
        tester = (
            Tester(module.eval(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_"])
        )
        if folded:
            # XNNPACK can also run the computation and take a dynamic weight.
            lowered = (
                tester.get_artifact().exported_program().graph_module.lowered_module_0
            )
            self.assertCountEqual(
                lowered.named_data_store_output.buffers,
                [tensor.detach().numpy().tobytes() for tensor in folded],
            )
        tester.to_executorch().serialize().run_method_and_compare_outputs()

    def test_fp32_weight_norm_conv1d(self):
        self._test_fully_delegated(self.WeightNormConv1d(), (torch.randn(1, 8, 16),))

    def test_fp32_computed_linear_weight_and_bias(self):
        module = self.ComputedLinear()
        self._test_fully_delegated(
            module, (torch.randn(2, 8),), folded=(module.weight * 2, module.bias + 1)
        )

    def test_early_noop_then_computed_linear_folding(self):
        module = self.ComputedLinear().eval()
        inputs = (torch.randn(2, 8),)
        program = torch.export.export(module, inputs)
        transformed = FoldConstantSubgraphsPass()(program)
        self.assertIs(transformed, program)
        targets = {node.target for node in transformed.graph.nodes}
        self.assertIn(torch.ops.aten.mul.Tensor, targets)
        self.assertIn(torch.ops.aten.add.Tensor, targets)
        self._test_fully_delegated(
            module, inputs, folded=(module.weight * 2, module.bias + 1)
        )

    def test_both_folding_stages_contribute(self):
        class SharedBiases(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 8))
                self.other_weight = torch.nn.Parameter(torch.randn(4, 8))
                self.bias = torch.nn.Parameter(torch.randn(4))

            def forward(self, x):
                bias = torch.cat((self.bias, self.bias)).split(4)[0]
                return F.linear(x, self.weight * 2, bias) + F.linear(
                    x, self.other_weight * 3, bias
                )

        module = SharedBiases().eval()
        inputs = (torch.randn(2, 8),)
        program = torch.export.export(module, inputs)
        transformed = FoldConstantSubgraphsPass()(program)
        targets = [node.target for node in transformed.graph.nodes]
        self.assertNotIn(torch.ops.aten.cat.default, targets)
        self.assertNotIn(torch.ops.aten.split.Tensor, targets)
        self.assertEqual(targets.count(torch.ops.aten.mul.Tensor), 2)
        transformed.validate()
        torch.testing.assert_close(transformed.module()(*inputs), module(*inputs))
        self._test_fully_delegated(
            module,
            inputs,
            folded=(module.weight * 2, module.other_weight * 3, module.bias),
        )

    def test_fp32_folded_name_does_not_replace_param(self):
        module = self.FoldedNameCollision()
        self._test_fully_delegated(
            module,
            (torch.randn(2, 8),),
            folded=(module.weight * 2, module.aten_mul_tensor_folded),
        )

    def test_quantized_computed_linear_bias(self):
        for is_dynamic, atol in ((False, 1e-3), (True, 5e-2)):
            with self.subTest(is_dynamic=is_dynamic):
                (
                    Tester(self.ComputedBiasLinear().eval(), (torch.randn(2, 8),))
                    .quantize(
                        Quantize(
                            quantization_config=get_symmetric_quantization_config(
                                is_per_channel=True, is_dynamic=is_dynamic
                            )
                        )
                    )
                    .export()
                    .to_edge_transform_and_lower()
                    .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                    .check_not(
                        [
                            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                            "executorch_exir_dialects_edge__ops_aten_linear_default",
                        ]
                    )
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs(atol=atol)
                )

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
        (
            Tester(self.ScaledAddmm().eval(), (torch.randn(2, 8),))
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

    def test_fp32_gemm_rejected_by_prior_partitioner(self):
        # The first partitioner tags the weight computation while checking the
        # addmm, rejects the addmm for its bias and delegates the mul. The second
        # one takes the addmm with a dynamic weight.
        (
            Tester(self.InputScaledBiasAddmm().eval(), (torch.randn(2, 8),))
            .export()
            .to_edge_transform_and_lower(
                ToEdgeTransformAndLower(
                    [
                        XnnpackPartitioner(configs=[AddmmConfig, MulConfig]),
                        XnnpackPartitioner(
                            force_non_static_weights_for_f32_linear=True
                        ),
                    ]
                )
            )
            .check_not(["executorch_exir_dialects_edge__ops_aten_addmm_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_custom_op_weight_not_folded(self):
        (
            Tester(self.CustomOpWeight().eval(), (torch.randn(2, 8),))
            .export()
            .to_edge_transform_and_lower()
            .check(["executorch_exir_dialects_edge__ops_aten_mm_default"])
        )

    def test_fp32_mutated_buffer_source_not_folded(self):
        # The program holds no initial data for a mutated buffer, so it isn't run.
        (
            Tester(self.MutatedBufferSource().eval(), (torch.randn(1, 4, 8),))
            .export()
            .to_edge_transform_and_lower()
            .check(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
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
        ):
            with self.subTest(module=type(module).__name__):
                (
                    Tester(module.eval(), inputs)
                    .export()
                    .to_edge_transform_and_lower()
                    .check([f"executorch_exir_dialects_edge__ops_{gemm}"])
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs()
                )

    def test_fp32_externally_stored_weight_not_folded(self):
        module = self.WeightNormConv1d().eval()
        inputs = (torch.randn(1, 8, 16),)
        tester = Tester(module, inputs)
        for node in tester.export().get_artifact().graph.nodes:
            if node.op == "placeholder" and node.name.startswith("p_"):
                node.meta["custom"] = {"delegate_constant_tag": "weights.ptd"}
        program = (
            tester.to_edge_transform_and_lower()
            .check(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .to_executorch()
            .get_artifact()
        )
        # The serialize stage would run the PTE without its data file.
        executorch_module = _load_for_executorch_from_buffer(
            program.buffer, bytes(program._tensor_data["weights.ptd"])
        )
        torch.testing.assert_close(
            executorch_module.run_method("forward", inputs)[0],
            module(*inputs),
            atol=1e-3,
            rtol=1e-3,
        )
