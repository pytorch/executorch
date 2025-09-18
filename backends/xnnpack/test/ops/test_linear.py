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

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)
from executorch.backends.xnnpack.test.tester import Quantize, Tester
from executorch.backends.xnnpack.test.tester.tester import (
    Partition,
    ToEdgeTransformAndLower,
)

from torch.export.graph_signature import ExportGraphSignature, InputKind

try:
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.quant_api import (
        Int8DynamicActivationIntxWeightConfig,
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

    def get_inputs(self, rank=3):
        # rank = 3 as default to inflate the act rank by 1 in batch dim
        # This is to make sure we don't specialize on 2D shapes.
        inp = torch.randn(self.in_size, self.ic).to(self.op_dtype)
        for _ in range(rank - 2):
            inp = inp.unsqueeze(0)
        assert inp.ndim == rank
        return (inp,)


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


class ParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1_weight = torch.nn.Parameter(torch.rand(output_size, input_size))
        self.linear1_bias = torch.nn.Parameter(torch.rand(output_size))

        self.linear2_weight = torch.nn.Parameter(torch.rand(output_size, input_size))
        self.linear2_bias = torch.nn.Parameter(torch.rand(output_size))

    def forward(self, x, y):
        a = torch.nn.functional.linear(x, self.linear1_weight, self.linear1_bias)
        b = torch.nn.functional.linear(y, self.linear2_weight, self.linear2_bias)
        return a + b


class SharedDQChain(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1_weight = torch.nn.Parameter(torch.rand(output_size, input_size))
        self.linear1_bias = torch.nn.Parameter(torch.rand(output_size))

        self.linear2_weight = torch.nn.Parameter(torch.rand(output_size, input_size))
        self.linear2_bias = torch.nn.Parameter(torch.rand(output_size))

    def forward(self, x):
        a = torch.nn.functional.linear(x, self.linear1_weight, self.linear1_bias)
        b = torch.nn.functional.linear(x, self.linear2_weight, self.linear2_bias)
        return a + b


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

    def setUp(self):
        torch._dynamo.reset()

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

    def _test_linear(
        self,
        make_module,
        uses_bias,
        num_batch_dims=1,
        quant_type=None,
        dtype: torch.dtype = torch.float,
        atol=1e-03,  # TODO(T212995726): Investigate right atol for rand[n] inputs
    ):
        """
        Helper function to test linear op with different configurations.
        """
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
        atol=5e-02,  # TODO(T212995726): Investigate right atol for rand[n] inputs
    ):
        """
        Helper function to test dynamic quantized linear op with different configurations.
        """
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
        atol: float = 5e-3,  # TODO(T212995726): Investigate right atol for rand[n] inputs
        rtol: float = 5e-3,  # TODO(T212995726): Investigate right rtol for rand[n] inputs
    ):
        """
        Helper function to test groupwise dynamic quantized linear op with different configurations.
        """
        quantize_(
            mod,
            Int8DynamicActivationIntxWeightConfig(
                # pyre-ignore[16]
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(group_size),
            ),
        )
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
                    "torch.ops.torchao.choose_qparams_affine.default": 1 * num_linears,
                    "torch.ops.torchao.quantize_affine.default": 1 * num_linears,
                    "torch.ops.torchao.dequantize_affine.default": 2 * num_linears,
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
        atol: float = 1e-03,  # TODO(T212995726): Investigate right atol for rand[n] inputs
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

    def test_qd8_f32_per_channel_shared_dq_chain(self):
        for use_bias in (False, True):
            module = SharedDQChain(
                input_size=13,
                output_size=17,
            )
            inputs = (torch.randn(1, 2, 13),)

            self._test_dqlinear(
                module,
                inputs,
                dynamic_shapes=None,
                is_per_channel=True,
                linear_count=2,
                uses_bias=use_bias,
            )

    def _test_qd8_per_channel_linear(self, dtype: torch.dtype = torch.float):
        for uses_bias in (False, True):
            module = BaseLinear(
                in_size=8,
                input_channels=13,
                output_channels=17,
                dtype=dtype,
                use_bias=uses_bias,
            )
            inputs = module.get_inputs()

            self._test_dqlinear(
                module,
                inputs,
                dynamic_shapes=({1: torch.export.Dim("batch", max=100)},),
                is_per_channel=True,
                uses_bias=uses_bias,
            )

    def _test_qd8_linear_per_tensor_unsupported(self, dtype: torch.dtype = torch.float):
        for uses_bias in (False, True):
            module = BaseLinear(
                in_size=8,
                input_channels=13,
                output_channels=17,
                dtype=dtype,
                use_bias=uses_bias,
            )
            inputs = module.get_inputs()
            dynamic_shapes = ({1: torch.export.Dim("batch", max=100)},)

            quant_config = get_symmetric_quantization_config(
                is_per_channel=False,
                is_dynamic=True,
            )

            for legacy_partitioner in (True, False):
                for per_op_mode in (True, False):
                    # Every combination should fail to partition Linear or [add]mm.
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
                        # should have [add]mm node
                        if uses_bias:
                            tester.check(
                                [
                                    "executorch_exir_dialects_edge__ops_aten_addmm_default",
                                ]
                            )
                        else:
                            tester.check(
                                [
                                    "executorch_exir_dialects_edge__ops_aten_mm_default",
                                ]
                            )
                    else:
                        tester.to_edge_transform_and_lower(
                            ToEdgeTransformAndLower([DynamicallyQuantizedPartitioner])
                        )
                        # should not have a delegate node
                        tester.check_not(
                            [
                                "torch.ops.higher_order.executorch_call_delegate",
                            ]
                        )
                    # No need to run the model, since it should fail to partition.
                    return

    def _test_qd8_per_channel_4w_linear(self, dtype: torch.dtype = torch.float):
        qconfig = self._get_4b_dqconfig()
        input_channels = [2, 63]
        output_channels = [1, 127]
        batches = [
            2,
        ]
        use_bias = [False, True]
        dtypes = [
            dtype,
        ]

        for bs, bias, ipc, opc, dtype in product(
            batches,
            use_bias,
            input_channels,
            output_channels,
            dtypes,
        ):
            module = BaseLinear(
                in_size=bs,
                input_channels=ipc,
                output_channels=opc,
                dtype=dtype,
                use_bias=bias,
            )
            inputs = module.get_inputs()

            self._test_dqlinear(
                module,
                inputs,
                dynamic_shapes=({1: torch.export.Dim("batch", max=100)},),
                is_per_channel=True,
                uses_bias=bias,
                qconfig=qconfig,
                atol=5e-2,  # TODO(T212995726): Investigate right atol for rand[n] inputs
            )

    def _test_qd8_per_token_weight_per_channel_group_int4(
        self, dtype: torch.dtype = torch.float
    ):
        M_sizes = [1, 2, 17, 31]
        K_sizes = [32, 32, 64, 128]
        bl_sizes = [32, 32, 32, 64]
        N_sizes = [2, 17, 92, 128]

        for input_rank in range(2, 4):
            for use_bias in [True, False]:
                for M, K, bl, N in zip(M_sizes, K_sizes, bl_sizes, N_sizes):
                    lin_mod = BaseLinear(
                        in_size=M,
                        input_channels=K,
                        output_channels=N,
                        dtype=dtype,
                        use_bias=use_bias,
                    )

                    inputs = lin_mod.get_inputs(rank=input_rank)
                    # Half requires slightly higher atol, but if you look at error it is not that bad:
                    # Difference: max: 0.00140380859375, abs: 0.00140380859375, mean abs error: 0.00042724609375.
                    # -- Model vs. Reference --
                    # Numel: 4, 4
                    # Median: -0.05023193359375, -0.0516357421875
                    # Mean: 0.2373046875, 0.237060546875
                    # Max: 1.0078125, 1.0078125
                    # Min: -0.08465576171875, -0.08441162109375
                    atol = (
                        1e-2 if dtype == torch.half else 5e-3
                    )  # TODO(T212995726): Investigate right atol for rand[n] inputs
                    self._test_groupwise_dq_linear(
                        lin_mod, inputs, group_size=bl, use_bias=use_bias, atol=atol
                    )

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
                    atol=5e-3,  # TODO(T212995726): Investigate right atol for rand[n] inputs
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

    # Tests for q[dp]8-f16-qc8w
    def test_qd8_f16_per_channel_linear(self):
        self._test_qd8_per_channel_linear(dtype=torch.half)

    def test_qd8_f16_per_tensor_linear(self):
        """
        XNNPACK doesn't support per_tensor quantized weights for dynamic quantized linear op.
        This test is to verify that we can't lower per_tensor quantized weights to per_channel quantized weights.
        """
        self._test_qd8_linear_per_tensor_unsupported(dtype=torch.half)

    # Tests for q[dp]8-f32-qc8w
    def test_qd8_f32_per_channel_linear(self):
        self._test_qd8_per_channel_linear(dtype=torch.float)

    def test_qd8_f32_per_tensor_linear(self):
        """
        XNNPACK doesn't support per_tensor quantized weights for dynamic quantized linear op.
        This test is to verify that we can't lower per_tensor quantized weights to per_channel quantized weights.
        """
        self._test_qd8_linear_per_tensor_unsupported(dtype=torch.half)

    # Tests for q[dp]8-f16-qc4w
    def test_linear_qd8_f16_per_channel_int4(self):
        self._test_qd8_per_channel_4w_linear(dtype=torch.half)

    # Tests for q[dp]8-f32-qc4w
    def test_linear_qd8_f32_per_channel_int4(self):
        self._test_qd8_per_channel_4w_linear(dtype=torch.float)

    # Tests for q[dp]8-f16-qb4w
    @unittest.skipIf(
        not torchao_installed, "Per Channel Group Quantization Required TorchAO"
    )
    def test_linear_qd8_f16_per_token_weight_per_channel_group_int4(self):
        self._test_qd8_per_token_weight_per_channel_group_int4(dtype=torch.half)

    # Tests for q[dp]8-f32-qb4w
    @unittest.skipIf(
        not torchao_installed, "Per Channel Group Quantization Required TorchAO"
    )
    def test_linear_qd8_f32_per_token_weight_per_channel_group_int4(self):
        self._test_qd8_per_token_weight_per_channel_group_int4(dtype=torch.float)

    @unittest.skipIf(
        not torchao_installed, "Per Channel Group Quantization Required TorchAO"
    )
    def test_linear_qd8_per_token_groupwise_unsupported_groupsize(self):
        # groupsize must be multiple of 32
        for dtype in [torch.float, torch.half]:
            lin_mod = BaseLinear(
                in_size=1,
                input_channels=60,
                output_channels=60,
                dtype=dtype,
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

    def test_qd8_per_channel_linear_parallel(self):
        in_size = 2
        input_size = 4
        output_size = 5
        for dtype in torch.float, torch.half:
            inputs = (
                torch.rand(in_size, input_size, dtype=dtype),
                torch.rand(in_size, input_size, dtype=dtype),
            )
            batch_dim = torch.export.Dim("batch", max=100)
            dynamic_shapes = ({0: batch_dim}, {0: batch_dim})

            self._test_dqlinear(
                ParallelLinear(input_size=input_size, output_size=output_size).to(
                    dtype
                ),
                inputs,
                dynamic_shapes=dynamic_shapes,
                linear_count=2,
                is_per_channel=True,
                uses_bias=True,
            )

    def test_qd8_per_channel_linear_with_two_batch(self):
        in_size = 2
        input_size = 14
        output_size = 15

        for dtype in torch.float, torch.half:
            for use_bias in (False, True):
                linear = BaseLinear(
                    in_size=in_size,
                    input_channels=input_size,
                    output_channels=output_size,
                    dtype=dtype,
                    use_bias=use_bias,
                )
                # Create inputs with two batch dimensions, i.e. 3D activation
                inputs = (torch.randn(in_size, in_size, input_size).to(dtype),)
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
            atol=1e-1,  # TODO(T212995726): Investigate right atol for rand[n] inputs
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
            atol=1e-1,  # TODO(T212995726): Investigate right atol for rand[n] inputs
        )

    def test_linear_qs8_as_fp32(self):
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

    def test_linear_qc8_as_fp32(self):
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

    def test_linear_qd8_as_fp32(self):
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

    def test_linear_with_force_non_static_weights_for_f32_linear(self):
        def check_signature(
            signature: ExportGraphSignature,
            force_flag: bool,
            use_bias: bool,
            legacy_mode: bool,
        ):
            num_params = 0
            if force_flag:
                num_params = 1  # weight_param
                if use_bias:
                    num_params += 1  # bias_param
            sign_params: int = 0
            input_specs = signature.input_specs
            for input_spec in input_specs:
                if input_spec.kind == InputKind.PARAMETER:
                    sign_params += 1
            assert (
                sign_params == num_params
            ), f"Expected {num_params} params, got {sign_params} with force_flag={force_flag}, use_bias={use_bias}, legacy_mode={legacy_mode}"

        for force_flag in (True, False):
            for use_bias in (True, False):
                for legacy_mode in (True, False):
                    module = BaseLinear(
                        in_size=8,
                        input_channels=13,
                        output_channels=17,
                        use_bias=use_bias,
                    )
                    inputs = module.get_inputs()
                    tester = Tester(module, inputs).export()
                    partitioner = XnnpackPartitioner(
                        force_non_static_weights_for_f32_linear=force_flag
                    )
                    if legacy_mode:
                        tester.to_edge()
                        partitioner_stage = Partition(partitioner=partitioner)
                        tester.partition(partition_stage=partitioner_stage)
                        tester.check_not(
                            [
                                (
                                    "executorch_exir_dialects_edge__ops_aten_mm_default"
                                    if use_bias
                                    else "executorch_exir_dialects_edge__ops_aten_addmm_default"
                                )
                            ]
                        )
                    else:
                        to_edge_and_transform_stage = ToEdgeTransformAndLower(
                            partitioners=[partitioner]
                        )
                        tester.to_edge_transform_and_lower(
                            to_edge_and_transform_stage=to_edge_and_transform_stage
                        )
                        tester.check_not(
                            ["executorch_exir_dialects_edge__ops_aten_linear_default"]
                        )

                    signature: ExportGraphSignature = (
                        tester.get_artifact().exported_program().graph_signature
                    )
                    check_signature(signature, force_flag, use_bias, legacy_mode)

                    tester.to_executorch()
                    tester.serialize()
                    tester.run_method_and_compare_outputs()
