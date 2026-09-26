# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack.operators.op_to_copy import (
    sort_decomposed_operations,
    ToCopyOperation,
)
from executorch.backends.xnnpack.partition.config.generic_node_configs import (
    ToCopyConfig,
)
from executorch.backends.xnnpack.test.tester import Tester, ToEdgeTransformAndLower
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir.dialects._ops import ops as exir_ops


class TestChannelsLastTaggedReshapePass(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_transpose_cast_order_uses_smaller_dtype_for_transpose(self):
        ops = [ToCopyOperation.TRANSPOSE, ToCopyOperation.CAST]

        sort_decomposed_operations(ops, torch.float16, torch.float32)

        self.assertEqual(ops, [ToCopyOperation.TRANSPOSE, ToCopyOperation.CAST])

        sort_decomposed_operations(ops, torch.float32, torch.float16)

        self.assertEqual(ops, [ToCopyOperation.CAST, ToCopyOperation.TRANSPOSE])

    def test_to_copy_config_rejects_integer_dtype_conversion(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randint(0, 10, (1, 3, 6, 6), dtype=torch.int32)
        to_copy = graph.call_function(
            exir_ops.edge.aten._to_copy.default,
            (x,),
            {"dtype": torch.float32, "memory_format": torch.channels_last},
        )
        to_copy.meta["val"] = torch.randn(1, 3, 6, 6).to(
            memory_format=torch.channels_last
        )

        self.assertFalse(ToCopyConfig().check_constraints(to_copy, None))

    def run_tester(self, module, inputs, skip_dim_order=False):
        tester = Tester(
            module.eval(),
            inputs,
        )
        tester = tester.export()
        if skip_dim_order:
            tester = tester.to_edge_transform_and_lower(
                ToEdgeTransformAndLower(
                    edge_compile_config=get_xnnpack_edge_compile_config(
                        skip_dim_order=True
                    )
                )
            )
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        else:
            tester = tester.to_edge_transform_and_lower()

        tester.check_not(
            ["executorch_exir_dialects_edge__ops_aten__to_copy_default"]
        ).to_executorch().serialize().run_method_and_compare_outputs()

    class ChannelLastBeforeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            y = x.to(memory_format=torch.channels_last)
            return self.linear(y)

    ChannelLastBeforeLinearModule = ChannelLastBeforeLinear()

    def test_channel_last_before_linear(self):
        self.run_tester(self.ChannelLastBeforeLinearModule, (torch.randn(1, 3, 3, 3),))

    class ContiguousBeforeConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = x.to(memory_format=torch.contiguous_format)
            return self.conv(y)

    ContiguousBeforeConvModule = ContiguousBeforeConv()

    def test_contiguous_before_conv(self):
        self.run_tester(self.ContiguousBeforeConvModule, (torch.randn(1, 3, 6, 6),))

    class DtypeAndMemoryFormatConversion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = x.to(torch.float, memory_format=torch.channels_last)
            return self.conv(y)

    DtypeAndMemoryFormatConversionModule = DtypeAndMemoryFormatConversion()

    def test_dtype_and_memory_format_conversion(self):
        self.run_tester(
            self.DtypeAndMemoryFormatConversionModule,
            (torch.randn(1, 3, 6, 6, dtype=torch.float16),),
        )

    class DtypeAndMemoryFormatWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            y = x.to(torch.float, memory_format=torch.channels_last)
            return self.linear(y)

    DtypeAndMemoryFormatWithLinearModule = DtypeAndMemoryFormatWithLinear()

    def test_dtype_and_memory_format_with_linear(self):
        self.run_tester(
            self.DtypeAndMemoryFormatWithLinearModule,
            (torch.randn(1, 3, 3, 3, dtype=torch.float16),),
        )

    def test_integer_to_float_to_copy_does_not_partition(self):
        to_dim_order_copy_name = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"
        (
            Tester(
                self.DtypeAndMemoryFormatConversionModule,
                (torch.randint(0, 10, (1, 3, 6, 6), dtype=torch.int32),),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_count({to_dim_order_copy_name: 1})
        )

    class DtypeOnlyConversion(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float)

    DtypeOnlyConversionModule = DtypeOnlyConversion()

    def test_dtype_only_conversion_with_skip_dim_order(self):
        self.run_tester(
            self.DtypeOnlyConversionModule,
            (torch.randn(1, 3, 3, 3, dtype=torch.float16),),
            skip_dim_order=True,
        )

    def test_memory_format_conversion_with_skip_dim_order(self):
        self.run_tester(
            self.ChannelLastBeforeLinearModule,
            (torch.randn(1, 3, 3, 3),),
            skip_dim_order=True,
        )

    def test_dtype_and_memory_format_conversion_with_skip_dim_order(self):
        self.run_tester(
            self.DtypeAndMemoryFormatConversionModule,
            (torch.randn(1, 3, 6, 6, dtype=torch.float16),),
            skip_dim_order=True,
        )

    class QuantizedToCopy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = self.conv(x)
            y = y.to(memory_format=torch.contiguous_format)
            return self.conv2(y)

    QuantizedToCopyModule = QuantizedToCopy()

    def test_quantized_to_copy(self):
        tester = Tester(
            self.QuantizedToCopyModule.eval(),
            (torch.randn(1, 3, 9, 9),),
        )

        tester.quantize().export().to_edge_transform_and_lower().check_not(
            [
                "executorch_exir_dialects_edge__ops_aten__to_copy_default",
                "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            ]
        ).to_executorch().serialize().run_method_and_compare_outputs(qtol=1)
