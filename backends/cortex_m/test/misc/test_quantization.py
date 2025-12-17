# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.exir.dialects._ops import ops as exir_ops


class SharedQspecMulipleClusters(torch.nn.Module):
    """Three linear shared qspec clusters."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 2,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 8,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 8,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 4,
    }

    def forward(self, x):
        x1 = torch.clone(x)
        x2 = x1 + x1
        x3 = torch.clone(x2)
        x3 = torch.clone(x3)
        x3 = torch.clone(x3)
        x4 = x3 + x3
        x5 = torch.transpose(x4, 2, 1)
        return x5


class SharedQspecInputForkNonShared(torch.nn.Module):
    """Shared qspec cluster with an input fork with both inputs as non-shared qspecs."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_maximum_default": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 4,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_maximum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
    }

    def forward(self, x, y):
        z = torch.maximum(x, y)
        return torch.flatten(z)


class SharedQspecInputForkShared(torch.nn.Module):
    """Shared qspec cluster with an input fork with both inputs as shared qspecs."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 5,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 5,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
    }

    def forward(self, x, y):
        x = torch.clone(x)
        y = torch.permute(y, (0, 1, 3, 2))
        z = torch.minimum(x, y)
        return z


class SharedQspecInputForkXShared(torch.nn.Module):
    """Shared qspec cluster with an input fork with left input as shared qspec."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_maximum_default": 1,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 4,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_maximum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
    }

    def forward(self, x, y):
        x = torch.t_copy(x)
        z = torch.maximum(x, y)
        return z


class SharedQspecInputForkYShared(torch.nn.Module):
    """Shared qspec cluster with an input fork with right input as shared qspec."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 5,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 5,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
    }

    def forward(self, x, y):
        y = torch.clone(y)
        z = torch.minimum(x, y)
        return torch.squeeze(z)


class SharedQspecInputForkXConstant(torch.nn.Module):
    """Shared qspec cluster with an input fork with left input as global constant."""

    ops_before_transforms = {}
    ops_after_transforms = {}
    constant = torch.tensor(5.0)

    def forward(self, x):
        return torch.minimum(self.constant, x)


class SharedQspecInputForkYConstant(torch.nn.Module):
    """Shared qspec cluster with an input fork with left input as local constant."""

    ops_before_transforms = {}
    ops_after_transforms = {}

    def forward(self, x):
        return torch.maximum(x, torch.tensor(5.0))


class SharedQspecOutputForkNonShared(torch.nn.Module):
    """Shared qspec cluster with an output fork with both outputs as non-shared qspecs."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
    }

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        y = x + x
        return x, y


class SharedQspecOutputForkShared(torch.nn.Module):
    """Shared qspec cluster with an output fork with both outputs as shared qspecs."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 6,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 4,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
    }

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        y = torch.clone(x)
        z = torch.permute_copy(x, (0, 2, 1, 3))
        return y, z, x


class SharedQspecManyForks(torch.nn.Module):
    """Shared qspec cluster with a number of forks to testmore complex structures."""

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_maximum_default": 2,
        "executorch_exir_dialects_edge__ops_aten_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 9,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 6,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_maximum_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
    }

    def forward(self, x):
        x1 = torch.clone(x)
        x2 = torch.maximum(x, x1)
        x3 = torch.maximum(x, torch.t(x2))
        x4 = torch.minimum(x2, x3)

        return x4


class SharedQspecSurroundedQuantizedOp(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_maximum_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 5,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 4,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_maximum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
    }

    def forward(self, x):
        x1 = torch.clone(x)
        x2 = torch.add(x1, x1)
        x3 = torch.maximum(x1, x2)
        return x3


class SharedQspecSurroundedQuantizedOpConstant(torch.nn.Module):
    ops_before_transforms = {}
    ops_after_transforms = {}

    def forward(self, x):
        x1 = torch.clone(x)
        x2 = torch.add(x1, torch.ones(2, 2))
        x3 = torch.maximum(x1, x2)
        return x3


class SharedQspecSub(torch.nn.Module):
    ops_before_transforms = {}
    ops_after_transforms = {}

    def forward(self, x, y):
        return torch.clone(x - y)


test_cases = {
    "multiple_clusters": McuTestCase(
        SharedQspecMulipleClusters(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "input_fork_non_shared": McuTestCase(
        SharedQspecInputForkNonShared(),
        (ramp_tensor(-2, 2, (2, 3, 4)), ramp_tensor(-1, 3, (2, 3, 4))),
    ),
    "input_fork_shared": McuTestCase(
        SharedQspecInputForkShared(),
        (ramp_tensor(-2, 2, (2, 3, 4, 5)), ramp_tensor(-1, 3, (2, 3, 5, 4))),
    ),
    "input_fork_x_shared": McuTestCase(
        SharedQspecInputForkXShared(),
        (ramp_tensor(-2, 2, (3, 4)), ramp_tensor(-1, 3, (4, 3))),
    ),
    "input_fork_y_shared": McuTestCase(
        SharedQspecInputForkYShared(),
        (ramp_tensor(-2, 2, (2, 3, 4)), ramp_tensor(-1, 3, (2, 3, 4))),
    ),
    "input_fork_x_constant": McuTestCase(
        SharedQspecInputForkXConstant(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "input_fork_y_constant": McuTestCase(
        SharedQspecInputForkYConstant(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "surrounded_quantized_op": McuTestCase(
        SharedQspecSurroundedQuantizedOp(),
        (ramp_tensor(-128, 2, (2, 3, 4)),),
    ),
    "surrounded_quantized_op_constant": McuTestCase(
        SharedQspecSurroundedQuantizedOpConstant(),
        (ramp_tensor(-2, 2, (2, 2)),),
    ),
    "output_fork_non_shared": McuTestCase(
        SharedQspecOutputForkNonShared(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "output_fork_shared": McuTestCase(
        SharedQspecOutputForkShared(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "many_forks": McuTestCase(
        SharedQspecManyForks(),
        (ramp_tensor(-20, 2, (4, 4)),),
    ),
    "non-quantized_op": McuTestCase(
        SharedQspecSub(),
        (ramp_tensor(0, 10, (5, 5)), ramp_tensor(0, 1, (5, 5))),
    ),
}

xfails = {
    "surrounded_quantized_op_constant": "Numerical error since the add is forced to have non-correct qparams.",
    "non-quantized_op": "Non-quantized ops are not currently supported in SharedQspecQuantizer.",
}


@parametrize("test_case", test_cases, xfails=xfails)
def test_shared_qspec_quantizer(test_case):
    """
    Test that ops which does not change dynamic range are able to use int8 portable kernels.
    """
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )

    # Check that all nodes in the graph are in int8
    artifact = tester.get_artifact()
    for node in artifact.exported_program().module().graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == exir_ops.edge.cortex_m.dequantize_per_tensor.default:
            continue

        assert get_first_fake_tensor(node).dtype == torch.int8, f"{node.name}"
