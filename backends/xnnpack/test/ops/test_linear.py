# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import unittest

from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.test.tester import Quantize, Tester
from executorch.backends.xnnpack.test.tester.tester import (
    Partition,
    ToEdgeTransformAndLower,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig

try:
    from torchao.quantization.quant_api import (
        int8_dynamic_activation_int4_weight,
        quantize_,
    )
    from torchao.utils import unwrap_tensor_subclass

    torchao_installed = True
except:
    torchao_installed = False


# Pytorch Modules Used for Testing
class BaseLinear(torch.nn.Module):
    def __init__(
        self,
        in_size: int = 2,
        input_channels: int = 4,
        output_channels: int = 4,
        dtype: torch.dtype = torch.float,
        use_bias: bool = False,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            input_channels, output_channels, bias=use_bias
        ).to(dtype=dtype)

        self.ic = input_channels
        self.oc = output_channels

        assert dtype in [torch.float, torch.half], "Unsupported op dtype"
        self.op_dtype = dtype
        self.in_size = in_size

    def forward(self, x):
        return self.linear(x)

    def get_inputs(self):
        return (torch.randn(1, self.in_size, self.ic).to(self.op_dtype),)


class AddMMModule(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.mat = torch.nn.Parameter(torch.randn(in_size, out_size))
        self.bias = torch.nn.Parameter(torch.randn(1, out_size))

    def forward(self, x):
        return torch.addmm(self.bias, x, self.mat)


class LinearReluModule(torch.nn.Module):
    def __init__(self, in_size, out_size, use_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.linear = torch.nn.Linear(in_size, out_size, bias=use_bias).to(dtype=dtype)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))

    def get_inputs(self):
        return (torch.randn(1, self.in_size, self.ic).to(self.op_dtype),)


class LinearParallelSequentialModule(torch.nn.Module):
    def __init__(
        self,
        in_size=2,
        input_size=4,
        intermediate_size=5,
        output_size=3,
        dtype=torch.float,
    ):
        super().__init__()
        self.linear1_weight = torch.nn.Parameter(
            torch.rand(intermediate_size, input_size)
        )
        self.linear1_bias = torch.nn.Parameter(torch.rand(intermediate_size))

        self.linear2_weight = torch.nn.Parameter(
            torch.rand(intermediate_size, input_size)
        )
        self.linear2_bias = torch.nn.Parameter(torch.rand(intermediate_size))

        self.linear3_weight = torch.nn.Parameter(
            torch.rand(output_size, intermediate_size)
        )
        self.linear3_bias = torch.nn.Parameter(torch.rand(output_size))
        self.in_size = in_size
        self.input_size = input_size
        self.dtype = torch.float

    def forward(self, x, y):
        a = torch.nn.functional.linear(x, self.linear1_weight, self.linear1_bias)
        b = torch.nn.functional.linear(y, self.linear2_weight, self.linear2_bias)
        c = torch.nn.functional.linear(b, self.linear3_weight, self.linear3_bias)
        return (a, c)

    def get_inputs(self):
        return (
            torch.rand(self.in_size, self.input_size, dtype=self.dtype),
            torch.rand(self.in_size, self.input_size, dtype=self.dtype),
        )


class LinearSequential(torch.nn.Module):
    def __init__(
        self,
        in_size=2,
        input_size=4,
        intermediate_size=5,
        output_size=3,
        dtype=torch.float,
    ):
        super().__init__()
        self.linear1_weight = torch.nn.Parameter(
            torch.rand(intermediate_size, input_size)
        )
        self.linear1_bias = torch.nn.Parameter(torch.rand(intermediate_size))

        self.linear2_weight = torch.nn.Parameter(
            torch.rand(output_size, intermediate_size)
        )
        self.linear2_bias = torch.nn.Parameter(torch.rand(output_size))
        self.in_size = in_size
        self.input_size = input_size
        self.dtype = torch.float

    def forward(self, x):
        a = torch.nn.functional.linear(x, self.linear1_weight, self.linear1_bias)
        b = torch.nn.functional.linear(a, self.linear2_weight, self.linear2_bias)
        return b

    def get_inputs(self):
        return (torch.rand(self.in_size, self.input_size, dtype=torch.float),)


class TestLinear(unittest.TestCase):
    """
    Test Class for XNNPACK Linear Operators.

    Notes:
        - XNNPACK Does not support Per Tensor Quantized Weights with Dynamic Activations
        - XNNPACK Only supports Per-Token Activation, so Dynamic per-tensor Quantization
          As done by the default dynamic quantization flow does Per-Token Quantization
          Activation under the hood, where the torch.nn.Module is doing Per-Tensor Quantization
          on the Activation. This is sufficient because Per-Token Quantization on Activations
          should produce strictly better results compared to Per-Tensor Quantization
    """

    @staticmethod
    def _get_4b_dqconfig() -> QuantizationConfig:
        # Returns a QuantizationConfig for 4b dynamic quantization for XNNPACK.
        qconfig: QuantizationConfig = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
            weight_qmin=-8,
            weight_qmax=7,
        )
        return qconfig

    def test_fp16_linear(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    num_batch_dims=num_batch_dims,
                    uses_bias=use_bias,
                    dtype=torch.float16,
                    atol=5e-2,
                )

    def test_fp32_linear(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    uses_bias=use_bias,
                    num_batch_dims=num_batch_dims,
                )

    def test_qc8_linear(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    uses_bias=use_bias,
                    quant_type="per_channel",
                    num_batch_dims=num_batch_dims,
                )

    def test_fp32_addmm(self):
        # Note that the ConvertToLinear pass requires the weight matrix to be transposed.
        self._test_linear(
            lambda in_size, out_size: AddMMModule(in_size, out_size),
            uses_bias=True,
        )

    def test_fp32_linear_fused_relu(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: LinearReluModule(
                        in_size,
                        out_size,
                        use_bias,  # noqa
                    ),
                    uses_bias=use_bias,
                    num_batch_dims=num_batch_dims,
                )

    def test_qs8_linear_fused_relu(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: LinearReluModule(
                        in_size,
                        out_size,
                        use_bias,  # noqa
                    ),
                    num_batch_dims=num_batch_dims,
                    uses_bias=use_bias,
                    quant_type="per_tensor",
                )

    def test_qs8_linear(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    uses_bias=use_bias,
                    num_batch_dims=num_batch_dims,
                    quant_type="per_tensor",
                )

    def test_qd8_per_channel_linear(self):
        for uses_bias in (False, True):
            inputs = (torch.randn(2, 4),)
            module = torch.nn.Linear(4, 5, bias=uses_bias)

            self._test_dqlinear(
                module,
                inputs,
                dynamic_shapes=({0: torch.export.Dim("batch", max=100)},),
                is_per_channel=True,
                uses_bias=uses_bias,
            )

    def test_qd8_per_channel_4w_linear(self):
        qconfig = self._get_4b_dqconfig()
        input_channels = [2, 63]
        output_channels = [1, 8, 127]
        batches = [2, 2]
        use_bias = [False, True]

        for bs, bias, ipc, opc in product(
            batches,
            use_bias,
            input_channels,
            output_channels,
        ):
            inputs = (torch.rand(bs, ipc),)
            module = torch.nn.Linear(ipc, opc, bias=bias)

            self._test_dqlinear(
                module,
                inputs,
                dynamic_shapes=({0: torch.export.Dim("batch", max=100)},),
                is_per_channel=True,
                uses_bias=bias,
                qconfig=qconfig,
            )

    def test_qd8_per_channel_linear_parallel(self):
        in_size = 2
        input_size = 4
        output_size = 5

        class ParallelLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(output_size, input_size)
                )
                self.linear1_bias = torch.nn.Parameter(torch.rand(output_size))

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(output_size, input_size)
                )
                self.linear2_bias = torch.nn.Parameter(torch.rand(output_size))

            def forward(self, x, y):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    y, self.linear2_weight, self.linear2_bias
                )
                return a + b

        inputs = (
            torch.rand(in_size, input_size, dtype=torch.float),
            torch.rand(in_size, input_size, dtype=torch.float),
        )
        batch_dim = torch.export.Dim("batch", max=100)
        dynamic_shapes = ({0: batch_dim}, {0: batch_dim})

        self._test_dqlinear(
            ParallelLinear(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            linear_count=2,
            is_per_channel=True,
            uses_bias=True,
        )

    def test_qd8_per_channel_linear_with_two_batch(self):
        in_size = 2
        input_size = 4
        output_size = 5

        linear = torch.nn.Linear(input_size, output_size)
        inputs = (torch.randn(2, in_size, input_size, dtype=torch.float),)
        batch_dim = torch.export.Dim("batch", max=100)
        dynamic_shapes = ({0: batch_dim, 1: batch_dim},)

        self._test_dqlinear(
            linear,
            inputs,
            dynamic_shapes=dynamic_shapes,
            linear_count=1,
            is_per_channel=True,
            uses_bias=True,
        )

    def test_qd8_per_channel_linear_sequential(self):
        lin_mod = LinearSequential()
        inputs = lin_mod.get_inputs()
        dynamic_shapes = ({0: torch.export.Dim("batch", max=100)},)

        self._test_dqlinear(
            lin_mod,
            inputs,
            dynamic_shapes=dynamic_shapes,
            linear_count=2,
            is_per_channel=True,
            uses_bias=True,
            atol=1e-1,
        )

    def test_qd8_per_channel_linear_parallel_and_sequential(self):
        lin_mod = LinearParallelSequentialModule()
        inputs = lin_mod.get_inputs()
        dynamic_shapes = (
            {0: torch.export.Dim("batch", max=100)},
            {0: torch.export.Dim("batch2", max=100)},
        )

        self._test_dqlinear(
            lin_mod,
            inputs,
            dynamic_shapes=dynamic_shapes,
            linear_count=3,
            is_per_channel=True,
            uses_bias=True,
            atol=1e-1,
        )

    @unittest.skipIf(
        not torchao_installed, "Per Channel Group Quantization Required TorchAO"
    )
    def test_qd8_fp32_per_token_weight_per_channel_group_int4(self):
        M_sizes = [1, 2, 17, 31]
        K_sizes = [32, 32, 64, 128]
        bl_sizes = [32, 32, 32, 64]
        N_sizes = [2, 17, 92, 128]

        for use_bias in [True, False]:
            for M, K, bl, N in zip(M_sizes, K_sizes, bl_sizes, N_sizes):
                lin_mod = BaseLinear(
                    input_channels=K,
                    output_channels=N,
                    dtype=torch.float,
                    use_bias=use_bias,
                )

                inputs = (torch.randn(1, M, K),)
                self._test_groupwise_dq_linear(
                    lin_mod, inputs, group_size=bl, use_bias=use_bias
                )

    @unittest.skipIf(
        not torchao_installed, "Per Channel Group Quantization Required TorchAO"
    )
    def test_qd8_fp16_per_token_weight_per_channel_group_int4(self):
        M_sizes = [1, 2, 17, 31]
        K_sizes = [32, 32, 64, 128]
        bl_sizes = [32, 32, 32, 64]
        N_sizes = [2, 17, 92, 128]

        for use_bias in [True, False]:
            for M, K, bl, N in zip(M_sizes, K_sizes, bl_sizes, N_sizes):
                lin_mod = BaseLinear(
                    in_size=M,
                    input_channels=K,
                    output_channels=N,
                    dtype=torch.float16,
                    use_bias=use_bias,
                )

                inputs = lin_mod.get_inputs()
                # This requires slightly higher atol, but if you look at error it is not that bad:
                # Difference: max: 0.00140380859375, abs: 0.00140380859375, mean abs error: 0.00042724609375.
                # -- Model vs. Reference --
                # Numel: 4, 4
                # Median: -0.05023193359375, -0.0516357421875
                # Mean: 0.2373046875, 0.237060546875
                # Max: 1.0078125, 1.0078125
                # Min: -0.08465576171875, -0.08441162109375
                self._test_groupwise_dq_linear(
                    lin_mod, inputs, group_size=bl, use_bias=use_bias, atol=1e-2
                )

    @unittest.skipIf(
        not torchao_installed, "Per Channel Group Quantization Required TorchAO"
    )
    def test_qd8_fp32_per_token_groupwise_unsupported_groupsize(self):
        # groupsize must be multiple of 32
        lin_mod = BaseLinear(
            in_size=1,
            input_channels=60,
            output_channels=60,
            dtype=torch.float32,
            use_bias=True,
        )
        inputs = lin_mod.get_inputs()

        with self.assertRaisesRegex(
            AssertionError,
            "Delegation to XNNPACK requires group_size to be a multiple of 32, but got 30",
        ):
            self._test_groupwise_dq_linear(
                lin_mod, inputs, group_size=30, use_bias=False, atol=1e-2
            )

    def _test_linear(
        self,
        make_module,
        uses_bias,
        num_batch_dims=1,
        quant_type=None,
        dtype: torch.dtype = torch.float,
        atol=1e-03,
    ):
        edge_op = (
            "executorch_exir_dialects_edge__ops_aten_addmm_default"
            if uses_bias
            else "executorch_exir_dialects_edge__ops_aten_mm_default"
        )

        in_sizes = [3, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]

        quant_config = None
        if quant_type is not None:
            if quant_type == "per_channel":
                quant_config = get_symmetric_quantization_config(
                    is_per_channel=True,
                    is_dynamic=False,
                )
            elif quant_type == "per_tensor":
                quant_config = get_symmetric_quantization_config(
                    is_per_channel=False,
                    is_dynamic=False,
                )
            else:
                raise ValueError(f"Unsupported quant type {quant_type}")

        """
        Note that torch.nn.Linear maps to aten.mm.default (no bias) or aten.addmm.default (bias),
        which ares then transformed into aten.linear.default by the ConvertToLinear pass.
        """
        for i, _ in enumerate(in_sizes):
            torch._dynamo.reset()
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            input_shape = [in_size] * num_batch_dims + [input_size]

            module = make_module(input_size, output_size).eval().to(dtype)
            inputs = (torch.randn(input_shape).to(dtype),)
            dynamic_shape = {}
            for i in range(num_batch_dims):
                dynamic_shape[i] = torch.export.Dim(f"batch{i}", min=2, max=in_size)

            dynamic_shape = (dynamic_shape,)

            for legacy_mode in (True, False):
                tester = Tester(module, inputs, dynamic_shapes=dynamic_shape)

                if quant_config:
                    tester.quantize(Quantize(quantization_config=quant_config))

                tester.export()
                if quant_config:
                    tester.check(["torch.ops.quantized_decomposed"])

                if legacy_mode:
                    tester.to_edge()
                    tester.partition()
                else:
                    tester.to_edge_transform_and_lower()

                tester.check_count(
                    {"torch.ops.higher_order.executorch_call_delegate": 1}
                )
                tester.check_not([edge_op])

                if quant_config:
                    tester.check_not(
                        [
                            "executorch_exir_dialects_edge__ops_aten_mm_default",
                            "executorch_exir_dialects_edge__ops_aten_addmm_default",
                        ]
                    )

                tester.to_executorch()
                tester.serialize()
                tester.run_method_and_compare_outputs(
                    qtol=bool(quant_config), atol=atol
                )

    def _test_dqlinear(
        self,
        module,
        inputs,
        dynamic_shapes,
        linear_count=1,
        is_per_channel=False,
        uses_bias=False,
        qconfig: Optional[QuantizationConfig] = None,
        atol=5e-02,
    ):
        quant_config = qconfig or get_symmetric_quantization_config(
            is_per_channel=is_per_channel,
            is_dynamic=True,
        )
        for legacy_partitioner in (True, False):
            for per_op_mode in (True, False):
                DynamicallyQuantizedPartitioner = XnnpackPartitioner(
                    config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
                    per_op_mode=per_op_mode,
                )

                tester = Tester(module, inputs, dynamic_shapes=dynamic_shapes)
                tester.quantize(Quantize(quantization_config=quant_config))
                tester.export()

                if legacy_partitioner:
                    tester.to_edge()
                    tester.partition(Partition(DynamicallyQuantizedPartitioner))
                else:
                    tester.to_edge_transform_and_lower(
                        ToEdgeTransformAndLower([DynamicallyQuantizedPartitioner])
                    )
                tester.check_count(
                    {
                        "torch.ops.higher_order.executorch_call_delegate": (
                            linear_count if per_op_mode else 1
                        )
                    }
                )
                tester.check_not(
                    [
                        "executorch_exir_dialects_edge__ops_aten_mm_default",
                        "executorch_exir_dialects_edge__ops_aten_addmm_default",
                    ]
                )

                tester.to_executorch()
                tester.serialize()
                tester.run_method_and_compare_outputs(atol=atol)

    def _test_groupwise_dq_linear(
        self,
        mod: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        use_bias: bool = False,
        group_size: int = 8,
        num_linears: int = 1,
        atol: float = 5e-3,
        rtol: float = 5e-3,
    ):
        quantize_(mod, int8_dynamic_activation_int4_weight(group_size=group_size))
        unwrap_tensor_subclass(mod)
        DynamicallyQuantizedPartitioner = XnnpackPartitioner(
            config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
            per_op_mode=True,
        )
        tester = (
            Tester(mod, inputs)
            .export()
            .check_count(
                {
                    "torch.ops.quant.choose_qparams_affine.default": 1 * num_linears,
                    "torch.ops.quant.quantize_affine.default": 1 * num_linears,
                    "torch.ops.quant.dequantize_affine.default": 2 * num_linears,
                    "torch.ops.aten.linear.default": 1 * num_linears,
                }
            )
        )
        (
            tester.to_edge_transform_and_lower(
                ToEdgeTransformAndLower([DynamicallyQuantizedPartitioner])
            )
        )

        (
            tester.check_count(
                {
                    "torch.ops.higher_order.executorch_call_delegate": 1,
                }
            )
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_quant_choose_qparams_affine_default",
                    "executorch_exir_dialects_edge__ops_quant_quantize_affine_default",
                    "executorch_exir_dialects_edge__ops_quant_dequantize_affine_default",
                    "executorch_exir_dialects_edge__ops_aten_mm_default",
                    "executorch_exir_dialects_edge__ops_aten_addmm_default",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=atol, rtol=rtol)
        )

    def _test_linear_overwrite_precision(
        self,
        make_module: Callable[[int, int], torch.nn.Module],
        uses_bias: bool,
        quant_type: str,
        quant_node_checks: List[Dict[str, int]],
        atol: float = 1e-03,
    ):
        """
        This test is to test the overwrite precision of linear op.
        We will test partitioning, lowering, and running the quantized linear model as fp32 linear op.
        When using legacy_mode, we will test we don't partition [add]mm given,
        (1) We can't assume that weights are always static (non param).
        (2) Alternatively, when lowering [add]mm to xnn::bmm we can't support bias.
        (2)(a) Only lowering non-bias [add]mm, which is only exposed on legacy_path deemed low ROI.
        """

        in_sizes = [3, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]

        assert quant_type in ["per_tensor", "per_channel", "per_channel_dynamic"]
        per_channel = "per_channel" in quant_type
        dynamic = "dynamic" in quant_type
        quant_config = get_symmetric_quantization_config(
            is_per_channel=per_channel,
            is_dynamic=dynamic,
        )
        # Using FP32 partitioner for this quantized graph
        partitioner = XnnpackFloatingPointPartitioner()

        def get_qnode_checks(quant_node_checks, dialect):
            d = {}
            assert dialect in ["aten", "edge"]
            if dialect == "aten":
                d = {
                    f"torch.ops.quantized_decomposed.{op}": count
                    for op, count in quant_node_checks.items()
                }
            elif dialect == "edge":
                d = {
                    f"executorch.exir.dialects.edge._ops.quantized_decomposed.{op}".replace(
                        ".", "_"
                    ): count
                    for op, count in quant_node_checks.items()
                }
            assert len(d) == len(quant_node_checks)
            return d

        for i, _ in enumerate(in_sizes):
            torch._dynamo.reset()
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            input_shape = [in_size] + [input_size]
            module = make_module(input_size, output_size).eval()
            inputs = (torch.randn(input_shape),)

            addmm_op_str = (
                "executorch_exir_dialects_edge__ops_aten_addmm_default"
                if uses_bias
                else "executorch_exir_dialects_edge__ops_aten_mm_default"
            )
            linear_op_str = "executorch_exir_dialects_edge__ops_aten_linear_default"

            for legacy_mode in (True, False):
                tester = (
                    Tester(module, inputs)
                    .quantize(Quantize(quantization_config=quant_config))
                    .export()
                    .dump_artifact()
                    .check_count(get_qnode_checks(quant_node_checks, "aten"))
                )

                if legacy_mode:
                    tester.to_edge()
                    tester.partition(Partition(partitioner=partitioner))
                    # We don't expect [add]mm to be partitioned
                    tester.check([addmm_op_str])
                else:
                    tester.to_edge_transform_and_lower(
                        ToEdgeTransformAndLower(partitioners=[partitioner])
                    )
                    # We do expect linear to be partitioned
                    tester.check_not([linear_op_str])

                # For legacy mode, fp32 permute_copy gets partitioned. (just a side effect)
                # For new mode, fp32 linear gets partitioned.
                tester.check_count(
                    {"torch.ops.higher_order.executorch_call_delegate": 1}
                )

                # Typically, we would not see any quantized ops in the graph.
                # But here we shouldn't partition these.
                tester.check_count(get_qnode_checks(quant_node_checks, "edge"))

                # TODO: Need to figure out how to load quantized ops in pybindings.
                # tester.to_executorch()
                # tester.serialize()
                # tester.run_method_and_compare_outputs(
                #     qtol=bool(quant_config), atol=atol
                # )

    def test_qs8_as_fp32(self):
        for use_bias in (True, False):
            self._test_linear_overwrite_precision(
                lambda in_size, out_size: torch.nn.Linear(
                    in_size, out_size, bias=use_bias  # noqa
                ),
                use_bias,
                "per_tensor",
                quant_node_checks={
                    "quantize_per_tensor.default": 2,  # 1: act, 1: output
                    "dequantize_per_tensor.default": 3,  # 1: act, 1: weight, 1: output
                },
            )

    def test_qc8_as_fp32(self):
        for use_bias in (True, False):
            self._test_linear_overwrite_precision(
                lambda in_size, out_size: torch.nn.Linear(
                    in_size, out_size, bias=use_bias  # noqa
                ),
                use_bias,
                "per_channel",
                quant_node_checks={
                    "quantize_per_tensor.default": 2,  # 1: act, 1: output
                    "dequantize_per_tensor.default": 2,  # 1: act, 1: output
                    "dequantize_per_channel.default": 1,  # 1: weight
                },
            )

    def test_qd8_as_fp32(self):
        for use_bias in (True, False):
            self._test_linear_overwrite_precision(
                lambda in_size, out_size: torch.nn.Linear(
                    in_size, out_size, bias=use_bias  # noqa
                ),
                use_bias,
                "per_channel_dynamic",
                quant_node_checks={
                    "quantize_per_tensor.tensor": 1,  # 1: act
                    "dequantize_per_tensor.tensor": 1,  #  1: act
                    "dequantize_per_channel.default": 1,  # 1: weight
                },
            )
