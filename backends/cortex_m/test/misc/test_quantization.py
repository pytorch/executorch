# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.exir.dialects._ops import ops as exir_ops


class CortexMInheritAllOps(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x):
        # x shape: (1, 3, 4, 5)
        x = x + x
        x = torch.ops.aten.squeeze.default(x)  # Remove dim 0: (3, 4, 5)
        x = torch.ops.aten.unsqueeze.default(x, 0)  # Add dim at 0: (1, 3, 4, 5)
        x = torch.ops.aten.squeeze_copy.default(x)  # (3, 4, 5)
        x = torch.ops.aten.unsqueeze_copy.default(x, 0)  # (1, 3, 4, 5)
        x = torch.ops.aten.squeeze.dims(x, [0])  # (3, 4, 5)
        x = torch.ops.aten.squeeze_copy.dim(
            x, 0
        )  # Remove first dim if size 1, otherwise same
        x = torch.ops.aten.squeeze.dim(x, 0)  # Same
        x = torch.ops.aten.unbind.int(x, 0)[0]  # Unbind and take first: (4, 5)
        x = torch.ops.aten.reshape.default(x, (1, 4, 5, 1))  # (1, 4, 5, 1)
        x = torch.ops.aten.repeat.default(x, [1, 1, 1, 2])  # (1, 4, 5, 2)
        x = torch.ops.aten.view.default(x, (1, 4, 10))  # (1, 4, 10)
        target_shape = torch.zeros(1, 4, 10)
        x = torch.ops.aten.view_as.default(x, target_shape)  # (1, 4, 10)
        x = torch.ops.aten.view_copy.default(x, (1, 2, 20))  # (1, 2, 20)
        x = torch.ops.aten.unflatten.int(x, 2, [4, 5])  # (1, 2, 4, 5)
        x = torch.ops.aten.flatten.using_ints(x, 1, 3)  # (1, 40)
        x = torch.ops.aten.repeat_interleave.self_int(x, 2, 1)  # (1, 80)
        x = torch.ops.aten.expand_copy.default(x, (2, 80))  # (2, 80)
        x = torch.ops.aten.expand.default(x, (2, 80))  # (2, 80)
        x = torch.ops.aten.tile.default(x, [1, 1])  # (2, 80)
        x = torch.ops.aten.split.Tensor(x, 40, 1)[0]  # (2, 40)
        x = torch.ops.aten.split_with_sizes.default(x, [20, 20], 1)[0]  # (2, 20)
        x = torch.ops.aten.split_copy.Tensor(x, 10, 1)[0]  # (2, 10)
        x = torch.ops.aten.chunk.default(x, 2, 1)[0]  # (2, 5)
        x = torch.ops.aten.pad.default(x, [1, 1, 0, 0], "constant", 0.0)  # (2, 7)
        x = torch.ops.aten.select.int(x, 1, 0)  # (2,)
        x = torch.ops.aten.select_copy.int(x, 0, 0)  # scalar -> need to reshape
        x = torch.ops.aten.unsqueeze.default(x, 0)  # (1,)
        x = torch.ops.aten.unsqueeze.default(x, 1)  # (1, 1)
        x = torch.ops.aten.slice.Tensor(x, 0, 0, 1)  # (1, 1)
        x = torch.ops.aten.slice_copy.Tensor(x, 1, 0, 1)  # (1, 1)
        x = torch.ops.aten.reshape.default(x, (1, 1))  # Ensure shape for transpose
        x = torch.ops.aten.transpose.int(x, 0, 1)  # (1, 1)
        x = torch.ops.aten.transpose_copy.int(x, 0, 1)  # (1, 1)
        x = torch.ops.aten.t_copy.default(x)  # (1, 1)
        x = torch.ops.aten.contiguous.default(x)  # (1, 1)
        x = torch.ops.aten.permute.default(x, [1, 0])  # (1, 1)
        x = torch.ops.aten.permute_copy.default(x, [0, 1])  # (1, 1)
        x = torch.ops.aten.flip.default(x, [0])  # (1, 1)
        y = torch.zeros_like(x)
        x = torch.ops.aten.copy_.default(y, x)  # (1, 1)
        x = torch.ops.aten.clone.default(x)  # (1, 1)
        return x


class CortexMOnlyInheritOps(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x):
        return torch.permute(torch.clone(x), (0, 1, 3, 2))


class CortexMQuantizeNonSupportedSub(torch.nn.Module):
    ops_before_transforms = {}

    ops_after_transforms = {}

    def forward(self, x, y):
        return y - x


test_cases = {
    "all_ops": McuTestCase(
        CortexMInheritAllOps(),
        (ramp_tensor(0, 10, (1, 3, 4, 5)),),
    ),
    "only_inherit_ops": McuTestCase(
        CortexMOnlyInheritOps(),
        (ramp_tensor(0, 10, (1, 3, 4, 5)),),
    ),
}


@parametrize("test_case", test_cases)
def test_inherit_int8_dtype(test_case):
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


test_cases = {
    "sub": McuTestCase(
        CortexMQuantizeNonSupportedSub(),
        (ramp_tensor(0, 10, (1, 3, 4, 5)), ramp_tensor(0, 1, (1, 3, 4, 5))),
    ),
}


@pytest.mark.xfail(
    reason="Non handled ops which change dynamic range currently not supported."
)
@parametrize("test_case", test_cases)
def test_quantize_unsupported_op(test_case):
    """
    Test an op which does change dynamic range and which is not suported by CMSIS-NN. Currently not supported.
    """
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )
