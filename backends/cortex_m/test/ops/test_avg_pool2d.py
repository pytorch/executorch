# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops


class CortexMAvgPool2d(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(
        self, kernel_size, stride, padding=0, ceil_mode=False, count_include_pad=False
    ):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(
            kernel_size,
            stride,
            padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    def forward(self, x):  # noqa: D102
        return self.pool(x)


# Prepare test cases: simple 2x2 pool on 4x4, and 3x3 stride 1 on 3x3
test_cases = {
    "avgpool_2x2": McuTestCase(
        CortexMAvgPool2d(kernel_size=2, stride=2), (ramp_tensor(0, 15, (1, 1, 4, 4)),)
    ),
    "avgpool_3x3_s1": McuTestCase(
        CortexMAvgPool2d(kernel_size=3, stride=1, padding=1),
        (ramp_tensor(0, 8, (1, 1, 3, 3)),),
    ),
    # additional pooling configurations: padding, stride, ceil_mode, count_include_pad
    "avgpool_2x2_pad1": McuTestCase(
        CortexMAvgPool2d(kernel_size=2, stride=2, padding=1),
        (ramp_tensor(0, 24, (1, 1, 5, 5)),),
    ),
    "avgpool_3x3_s2_pad1": McuTestCase(
        CortexMAvgPool2d(kernel_size=3, stride=2, padding=1),
        (ramp_tensor(0, 15, (1, 1, 4, 4)),),
    ),
    "avgpool_3x3_s2_pad1_countinc": McuTestCase(
        CortexMAvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=True),
        (ramp_tensor(0, 15, (1, 1, 4, 4)),),
    ),
}


# ceil_mode=True is not supported by the CMSIS-NN avg_pool kernel; the convert
# pass leaves aten.avg_pool2d in the graph for a portable kernel to handle. The
# Cortex-M runner does not register aten.avg_pool2d, so this is dialect-only.
fallback_test_cases = {
    "avgpool_2x2_ceil_mode": McuTestCase(
        CortexMAvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
        (ramp_tensor(0, 24, (1, 1, 5, 5)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_avg_pool2d(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    ops_after = dict(test_case.model.ops_after_transforms)
    if test_case.model.pool.count_include_pad:
        ops_after["executorch_exir_dialects_edge__ops_cortex_m_pad_default"] = 1
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        ops_after,
        qtol=1,
    )

    from executorch.backends.cortex_m.library import cmsis_nn

    module = tester.get_artifact(StageType.RUN_PASSES).exported_program().module()
    pool_target = exir_ops.edge.cortex_m.quantized_avg_pool2d.default
    [pool_node] = [
        n
        for n in module.graph.nodes
        if n.op == "call_function" and n.target == pool_target
    ]
    scratch_arg = pool_node.args[-1]
    scratch_size = scratch_arg.args[0][0][0]

    input_node = pool_node.args[0]
    input_shape = input_node.meta["val"].shape
    output_shape = pool_node.meta["val"].shape
    expected_size = cmsis_nn.avgpool_buffer_size(
        cortex_m_target.backend,
        cmsis_nn.DataType.A8W8,
        dim_dst_width=int(output_shape[3]),
        ch_src=int(input_shape[1]),
    )
    assert (
        scratch_size == expected_size
    ), f"scratch buffer size mismatch: got {scratch_size}, expected {expected_size}"


@parametrize("test_case", fallback_test_cases)
def test_dialect_avg_pool2d_fallback(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        {
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        },
        {
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
            "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 2,
        },
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_avg_pool2d(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)
