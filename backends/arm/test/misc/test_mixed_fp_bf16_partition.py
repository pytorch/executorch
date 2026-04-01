# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    tosa_support_factory,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class MixedConv(torch.nn.Module):
    def __init__(self, weight_dtype: torch.dtype):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, bias=False)
        self.conv.weight = torch.nn.Parameter(self.conv.weight.to(weight_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CastInputConv(torch.nn.Module):
    def __init__(self, weight_dtype: torch.dtype):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, bias=False)
        self.conv.weight = torch.nn.Parameter(self.conv.weight.to(weight_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.to(torch.bfloat16))


class ParallelConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bf16 = torch.nn.Conv2d(3, 4, kernel_size=3, bias=False).to(
            dtype=torch.bfloat16
        )
        self.conv_fp32 = torch.nn.Conv2d(3, 4, kernel_size=3, bias=False).to(
            dtype=torch.float32
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conv_bf16(x), self.conv_fp32(y)


def test_mixed_fp32_bf16_inputs_rejected_no_target():
    test_data = (torch.randn(1, 3, 8, 8, dtype=torch.float32),)
    exported_program = to_edge(
        export(MixedConv(torch.bfloat16), test_data, strict=True),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    reporter = WhyNoPartitionReporter()
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP+bf16")
    support = tosa_support_factory(tosa_spec, exported_program, reporter)

    conv_node = exported_program.graph_module.graph.find_nodes(
        op="call_function", target=exir_ops.edge.aten.convolution.default
    )[0]

    assert support.is_node_supported(exported_program.graph_module, conv_node) is False
    assert "Mixed floating-point input" in reporter.get_table_report()


def test_mixed_bf16_cast_fp32_inputs_accepted_no_target():
    test_data = (torch.randn(1, 3, 8, 8, dtype=torch.float32),)
    exported_program = to_edge(
        export(CastInputConv(torch.bfloat16), test_data, strict=True)
    ).exported_program()
    reporter = WhyNoPartitionReporter()
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP+bf16")
    support = tosa_support_factory(tosa_spec, exported_program, reporter)

    conv_node = exported_program.graph_module.graph.find_nodes(
        op="call_function", target=exir_ops.edge.aten.convolution.default
    )[0]

    assert support.is_node_supported(exported_program.graph_module, conv_node) is True


def test_bf16_rejected_without_tosa_support_no_target():
    test_data = (torch.randn(1, 3, 8, 8, dtype=torch.bfloat16),)
    exported_program = to_edge(
        export(MixedConv(torch.bfloat16), test_data, strict=True)
    ).exported_program()
    reporter = WhyNoPartitionReporter()
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP")
    support = tosa_support_factory(tosa_spec, exported_program, reporter)

    conv_node = exported_program.graph_module.graph.find_nodes(
        op="call_function", target=exir_ops.edge.aten.convolution.default
    )[0]

    assert support.is_node_supported(exported_program.graph_module, conv_node) is False
    assert "Had torch.bfloat16 input" in reporter.get_table_report()


def test_parallel_bf16_fp32_inputs_accepted_no_target():
    test_data = (
        torch.randn(1, 3, 8, 8, dtype=torch.bfloat16),
        torch.randn(1, 3, 8, 8, dtype=torch.float32),
    )
    exported_program = to_edge(
        export(ParallelConv(), test_data, strict=True)
    ).exported_program()
    reporter = WhyNoPartitionReporter()
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP+bf16")
    support = tosa_support_factory(tosa_spec, exported_program, reporter)

    conv_nodes = exported_program.graph_module.graph.find_nodes(
        op="call_function", target=exir_ops.edge.aten.convolution.default
    )

    assert all(
        support.is_node_supported(exported_program.graph_module, conv_node) is True
        for conv_node in conv_nodes
    )
