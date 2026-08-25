# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Test the pad_constant_nd op which pads the input tensor at specific dimension(s).
#
from typing import Tuple

import torch
import torch.nn.functional as F
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    SymbolicShapeSupportCheck,
)
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.exir import to_edge
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export

aten_op = "torch.ops.aten.pad.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_pad_default"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    "4dim_last1dim": lambda: (
        torch.rand(1, 1, 16, 16),
        (1, 1, 0, 0, 0, 0, 0, 0),
        1,
        "constant",
    ),
    "4dim_last2dim": lambda: (
        torch.rand(1, 1, 16, 16),
        (1, 0, 1, 0, 0, 0, 0, 0),
        2,
        "constant",
    ),
    "4dim_last3dim": lambda: (
        torch.rand(1, 1, 16, 16),
        (1, 1, 0, 2, 0, 2, 0, 0),
        3,
        "constant",
    ),
    "4dim_last4dim": lambda: (
        torch.rand(1, 1, 16, 16),
        (1, 0, 1, 1, 0, 2, 0, 2),
        4,
        "constant",
    ),
    "3dim_last1dim": lambda: (torch.rand(1, 1, 16), (1, 1, 0, 0, 0, 0), 1, "constant"),
    "3dim_last2dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 1, 0, 0), 2, "constant"),
    "3dim_last3dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 0, 1, 1), 3, "constant"),
    "2dim_last1dim": lambda: (torch.rand(1, 1, 16), (1, 1, 0, 0), 1, "constant"),
    "2dim_last2dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 1), 2, "constant"),
    "4dim_reflect": lambda: (
        torch.rand(6, 6, 6, 6),
        (3, 3, 3, 3, 3, 3),
        None,
        "reflect",
    ),
    "4dim_replicate": lambda: (
        torch.rand(3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3),
        None,
        "replicate",
    ),
    "4dim_circular": lambda: (
        torch.rand(3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3),
        None,
        "circular",
    ),
    "2dim_reflect": lambda: (
        torch.rand(6, 6),
        (3, 3),
        None,
        "reflect",
    ),
    "2dim_replicate": lambda: (
        torch.rand(3, 3),
        (3, 3),
        None,
        "replicate",
    ),
    "2dim_circular": lambda: (
        torch.rand(3, 3),
        (3, 3),
        None,
        "circular",
    ),
}

test_data_suite_bf16 = {
    "4dim_last1dim_bf16": lambda: (
        torch.rand(1, 1, 8, 8, dtype=torch.bfloat16),
        (1, 1, 0, 0, 0, 0, 0, 0),
        1.0,
        "constant",
    ),
    "3dim_last1dim_bf16": lambda: (
        torch.rand(1, 1, 8, dtype=torch.bfloat16),
        (1, 0, 1, 0, 0, 0),
        -0.5,
        "constant",
    ),
}
test_data_suite_fp16 = {
    "4dim_last1dim_fp16": lambda: (
        torch.rand(1, 1, 8, 8, dtype=torch.float16),
        (1, 1, 0, 0, 0, 0, 0, 0),
        1.0,
        "constant",
    ),
    "3dim_last1dim_fp16": lambda: (
        torch.rand(1, 1, 8, dtype=torch.float16),
        (1, 0, 1, 0, 0, 0),
        -0.5,
        "constant",
    ),
}
test_data_suite_fp8 = {
    "4dim_last1dim_fp8e4m3": lambda: (
        torch.rand(1, 1, 8, 8, dtype=torch.float32).to(torch.float8_e4m3fn),
        (1, 1, 0, 0, 0, 0, 0, 0),
        1.0,
        "constant",
        "fp8e4m3",
    ),
    "3dim_last1dim_fp8e5m2": lambda: (
        torch.rand(1, 1, 8, dtype=torch.float32).to(torch.float8_e5m2),
        (1, 0, 1, 0, 0, 0),
        -0.5,
        "constant",
        "fp8e5m2",
    ),
}


class ConstantPadND(torch.nn.Module):
    def __init__(
        self,
        pad: Tuple,
        value: float | None = None,
        mode: str = "constant",
    ):
        super().__init__()
        self.value = value
        self.mode = mode
        nonzero_idx = len(pad)
        for i in range(0, len(pad), 2):
            if pad[i] + pad[i + 1] == 0:
                nonzero_idx = i
                break
        self.pad = pad[:nonzero_idx]

    def forward(self, x: torch.Tensor):
        return F.pad(x, pad=self.pad, mode=self.mode, value=self.value)


class RawConstantPadND(torch.nn.Module):
    def __init__(self, pad: Tuple, value: float = 0.0):
        super().__init__()
        self.pad = pad
        self.value = value

    def forward(self, x: torch.Tensor):
        return F.pad(x, pad=self.pad, mode="constant", value=self.value)


def _constant_pad_nd_node(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    dynamic_shapes=None,
) -> torch.fx.Node:
    edge = to_edge(
        export(module, example_inputs, dynamic_shapes=dynamic_shapes, strict=True)
    )
    return next(
        n
        for n in edge.exported_program().graph.nodes
        if n.target == exir_ops.edge.aten.constant_pad_nd.default
    )


def _is_tosa_without_shape_extension_supported(node: torch.fx.Node) -> bool:
    return SymbolicShapeSupportCheck(WhyNoPartitionReporter()).is_node_supported(
        {}, node
    )


def test_constant_pad_nd_no_target_u55_symbolic_padded_axis_not_delegated():
    input_tensor = torch.rand(1, 3, 8, 8, 5)
    width = Dim("width", min=4, max=10)
    node = _constant_pad_nd_node(
        RawConstantPadND((0, 1, 0, 0, 0, 0, 0, 0)),
        (input_tensor,),
        dynamic_shapes={"x": {4: width}},
    )

    assert not _is_tosa_without_shape_extension_supported(node)


def test_constant_pad_nd_no_target_u55_symbolic_unpadded_axis_not_delegated():
    input_tensor = torch.rand(1, 3, 8, 8, 5)
    width = Dim("width", min=4, max=10)
    node = _constant_pad_nd_node(
        RawConstantPadND((0, 0, 1, 0, 0, 0, 0, 0)),
        (input_tensor,),
        dynamic_shapes={"x": {4: width}},
    )

    assert not _is_tosa_without_shape_extension_supported(node)


def test_constant_pad_nd_no_target_u55_static_padded_axis_supported():
    input_tensor = torch.rand(1, 3, 8, 8, 5)
    node = _constant_pad_nd_node(
        RawConstantPadND((0, 1, 0, 0, 0, 0, 0, 0)),
        (input_tensor,),
    )

    assert _is_tosa_without_shape_extension_supported(node)


def test_constant_pad_nd_u55_INT_dynamic_padded_axis_not_delegated():
    input_tensor = torch.rand(1, 3, 8, 8, 5)
    width = Dim("width", min=4, max=10)
    tester = ArmTester(
        RawConstantPadND((0, 1, 0, 0, 0, 0, 0, 0)),
        (input_tensor,),
        common.get_u55_compile_spec(),
        dynamic_shapes=({4: width},),
    )

    tester.quantize().export().to_edge().partition()
    targets = {
        node.target
        for node in tester.stages[tester.cur].artifact.exported_program().graph.nodes
    }

    assert exir_ops.edge.aten.constant_pad_nd.default in targets
    assert torch.ops.higher_order.executorch_call_delegate not in targets


def test_constant_pad_nd_u85_INT_dynamic_padded_axis_not_delegated():
    input_tensor = torch.rand(1, 3, 8, 8, 5)
    width = Dim("width", min=4, max=10)
    tester = ArmTester(
        RawConstantPadND((0, 1, 0, 0, 0, 0, 0, 0)),
        (input_tensor,),
        common.get_u85_compile_spec(),
        dynamic_shapes=({4: width},),
    )

    tester.quantize().export().to_edge().partition()
    targets = {
        node.target
        for node in tester.stages[tester.cur].artifact.exported_program().graph.nodes
    }

    assert exir_ops.edge.aten.constant_pad_nd.default in targets
    assert torch.ops.higher_order.executorch_call_delegate not in targets


def test_constant_pad_nd_u55_INT_static_5d_padded_axis_delegated():
    input_tensor = torch.rand(1, 3, 8, 8, 5)
    tester = ArmTester(
        RawConstantPadND((0, 1, 0, 0, 0, 0, 0, 0)),
        (input_tensor,),
        common.get_u55_compile_spec(),
    )

    tester.quantize().export().to_edge_transform_and_lower()
    targets = {
        node.target
        for node in tester.stages[tester.cur].artifact.exported_program().graph.nodes
    }

    assert torch.ops.higher_order.executorch_call_delegate in targets


@common.parametrize(
    "test_data",
    test_data_suite | test_data_suite_bf16 | test_data_suite_fp16,
)
def test_constant_pad_nd_tosa_FP(test_data: Tuple):
    test_data, padding, value, mode = test_data()
    pipeline = TosaPipelineFP[input_t1](
        ConstantPadND(padding, value, mode),
        (test_data,),
        aten_op,
        exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_fp8)
def test_constant_pad_nd_tosa_FP_fp8(test_data: Tuple):
    test_data, padding, value, mode, tosa_extension = test_data()
    pipeline = TosaPipelineFP[input_t1](
        ConstantPadND(padding, value, mode),
        (test_data,),
        aten_op,
        exir_op,
        tosa_extensions=[tosa_extension],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_constant_pad_nd_tosa_INT(test_data: Tuple):
    test_data, padding, value, mode = test_data()
    pipeline = TosaPipelineINT[input_t1](
        ConstantPadND(padding, value, mode),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_constant_pad_nd_tosa_INT_a16w8(test_data: Tuple):
    """Test constant_pad_nd op with int16 I/O quantization for TOSA INT."""
    test_data, padding, value, mode = test_data()
    pipeline = TosaPipelineINT[input_t1](
        ConstantPadND(padding, value, mode),
        (test_data,),
        aten_op,
        exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize(
    "test_data", test_data_suite | test_data_suite_bf16 | test_data_suite_fp16
)
@common.SkipIfNoModelConverter
def test_constant_pad_nd_vgf_no_quant(test_data: Tuple):
    inp, padding, value, mode = test_data()
    pipeline = VgfPipeline[input_t1](
        ConstantPadND(padding, value, mode),
        (inp,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_constant_pad_nd_vgf_quant(test_data: Tuple):
    inp, padding, value, mode = test_data()
    pipeline = VgfPipeline[input_t1](
        ConstantPadND(padding, value, mode),
        (inp,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_constant_pad_nd_vgf_quant_a16w8(test_data: Tuple):
    inp, padding, value, mode = test_data()
    pipeline = VgfPipeline[input_t1](
        ConstantPadND(padding, value, mode),
        (inp,),
        aten_op,
        exir_op,
        quantize=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(get_symmetric_a16w8_quantization_config())
    pipeline.run()
