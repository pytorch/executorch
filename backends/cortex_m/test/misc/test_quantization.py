# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_TENSOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.backends.test.harness.stages import Quantize, StageType
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

    ops_before_transforms: dict[str, int] = {}
    ops_after_transforms: dict[str, int] = {}
    constant = torch.tensor(5.0)

    def forward(self, x):
        return torch.minimum(self.constant, x)


class SharedQspecInputForkYConstant(torch.nn.Module):
    """Shared qspec cluster with an input fork with left input as local constant."""

    ops_before_transforms: dict[str, int] = {}
    ops_after_transforms: dict[str, int] = {}

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
    ops_before_transforms: dict[str, int] = {}
    ops_after_transforms: dict[str, int] = {}

    def forward(self, x):
        x1 = torch.clone(x)
        x2 = torch.add(x1, torch.ones(2, 2))
        x3 = torch.maximum(x1, x2)
        return x3


class SharedQspecSub(torch.nn.Module):
    ops_before_transforms: dict[str, int] = {}
    ops_after_transforms: dict[str, int] = {}

    def forward(self, x, y):
        return torch.clone(x - y)


class SharedQspecCompetingQspecs(torch.nn.Module):
    ops_before_transforms: dict[str, int] = {}
    ops_after_transforms: dict[str, int] = {}

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.conv.weight.data = torch.tensor(
            [
                [[[0.1]], [[-0.2]], [[0.3]]],
                [[[-0.1]], [[0.2]], [[-0.3]]],
                [[[0.05]], [[-0.05]], [[0.15]]],
            ],
            dtype=self.conv.weight.dtype,
        )
        self.conv.bias.data = torch.tensor([0.0, 0.1, -0.1], dtype=self.conv.bias.dtype)

    def forward(self, x):
        return torch.cat([self.conv(x), x], dim=1)


class SharedQspecNoQspecs(torch.nn.Module):
    ops_before_transforms: dict[str, int] = {}
    ops_after_transforms: dict[str, int] = {}

    def forward(self, x):
        z = torch.clone(x - x)
        return z - z


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
}


@parametrize("test_case", test_cases)
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


float_test_cases = {
    "non-quantized_op": McuTestCase(
        SharedQspecSub(),
        (ramp_tensor(0, 10, (5, 5)), ramp_tensor(0, 1, (5, 5))),
    ),
    "competing_qspecs": McuTestCase(
        SharedQspecCompetingQspecs(),
        (ramp_tensor(0, 10, (1, 3, 5, 5)).to(memory_format=torch.channels_last),),
    ),
    "no_qspecs": McuTestCase(
        SharedQspecNoQspecs(),
        (ramp_tensor(0, 10, (1, 3, 5, 5)),),
    ),
}


@parametrize("test_case", float_test_cases)
def test_shared_qspec_quantizer_no_qspecs(test_case):
    """
    Test that ops which does not change dynamic range are able to use int8 portable kernels.
    """
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


def _conv_inputs():
    return (ramp_tensor(-2, 2, (1, 2, 6, 6)).to(memory_format=torch.channels_last),)


def _has_activation(targets):
    # An unfused quantized relu lowers to clamp.
    return any("relu" in t or "clamp" in t for t in targets)


@pytest.mark.parametrize(
    "configs, per_channel_weights",
    [
        ([INT8_PER_TENSOR_CONFIG], 0),
        ([INT8_PER_TENSOR_CONFIG, INT8_PER_CHANNEL_CONFIG], 1),
    ],
)
def test_set_node_target_overrides_earlier_configs(configs, per_channel_weights):
    # Convolutions default to per-channel weights; the last set_node_target
    # call decides.
    quantizer = CortexMQuantizer()
    for config in configs:
        quantizer.set_node_target(torch.ops.aten.conv2d.default, config)
    model = torch.nn.Conv2d(2, 4, 3, bias=False).eval()
    tester = CortexMTester(model, _conv_inputs())
    tester.quantize(Quantize(quantizer)).export().to_edge()
    tester.check_count(
        {
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": per_channel_weights
        }
    )


def test_set_node_target_keeps_fused_activation():
    # The override must still match conv + relu as one pattern, so the relu
    # folds into the convolution.
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3), torch.nn.ReLU()).eval()
    quantizer = CortexMQuantizer().set_node_target(
        torch.ops.aten.conv2d.default, INT8_PER_TENSOR_CONFIG
    )
    tester = CortexMTester(model, _conv_inputs())
    tester.quantize(Quantize(quantizer)).export().to_edge().run_passes()
    tester.run_method_and_compare_outputs(inputs=_conv_inputs(), qtol=1)
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    targets = [str(n.target) for n in program.graph.nodes if n.op == "call_function"]
    assert any("quantized_conv2d" in t for t in targets)
    assert not _has_activation(targets)


@pytest.mark.parametrize(
    "node_target, linear_quantized, relu_folded",
    [
        (torch.ops.aten.conv2d.default, True, True),
        (torch.ops.aten.relu.default, True, False),
        (torch.ops.aten.linear.default, False, False),
    ],
)
def test_set_node_target_none_leaves_node_target_unquantized(
    node_target, linear_quantized, relu_folded
):
    # Linear + relu fuses unless one of them is left unquantized; relu starts
    # no pattern, but None still applies to it.
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()).eval()
    quantizer = CortexMQuantizer().set_node_target(node_target, None)
    tester = CortexMTester(model, (ramp_tensor(-2, 2, (1, 4)),))
    tester.quantize(Quantize(quantizer)).export().to_edge().run_passes()
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    targets = [str(n.target) for n in program.graph.nodes if n.op == "call_function"]
    assert any("quantized_linear" in t for t in targets) == linear_quantized
    assert _has_activation(targets) != relu_folded


@pytest.mark.parametrize(
    "node_target, config, error, match",
    [
        (
            torch.ops.aten.conv2d.padding,
            INT8_PER_TENSOR_CONFIG,
            ValueError,
            "is not supported by the cortex-m backend",
        ),
        (torch.ops.aten.linear, None, TypeError, "expected an OpOverload"),
    ],
)
def test_set_node_target_rejects_invalid_target(node_target, config, error, match):
    with pytest.raises(error, match=match):
        CortexMQuantizer().set_node_target(node_target, config)
