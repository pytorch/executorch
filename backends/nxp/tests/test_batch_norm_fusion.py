# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
import pytest
import torch
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    AddMMConverter,
    MMConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.view_copy_converter import (
    ViewCopyConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    neutron_target_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    OverrideTargetSupportCheck,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import ConvBNModule
from torch import nn
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class BatchNormModule(torch.nn.Module):
    def __init__(self, input_rank: int, num_features: int, eps: float = 1e-5):
        super().__init__()
        match input_rank - 2:
            case 1:
                self.batch_norm = nn.BatchNorm1d(num_features, eps)
            case 2:
                self.batch_norm = nn.BatchNorm2d(num_features, eps)
            case 3:
                self.batch_norm = nn.BatchNorm3d(num_features, eps)
            case _:
                raise ValueError
        self.eval()

    def forward(self, x):
        return self.batch_norm(x)


class ConvBatchNormModule(torch.nn.Module):
    def __init__(
        self, bias: bool, input_rank: int, num_features: int, eps: float = 1e-5
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            bias=bias,
        )
        self.batch_norm = BatchNormModule(input_rank, num_features, eps)
        self.eval()

    def forward(self, x):
        x = self.conv(x)
        return self.batch_norm(x)


class LinearBatchNormModule(torch.nn.Module):
    def __init__(
        self,
        bias: bool,
        input_rank: int,
        fc_in_features: int,
        fc_out_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(fc_in_features, fc_out_features, bias=bias)
        self.batch_norm = BatchNormModule(input_rank, fc_out_features, eps)
        self.eval()

    def forward(self, x):
        x = self.linear(x)
        return self.batch_norm(x)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
@pytest.mark.parametrize(
    "input_shape", [[4, 6, 8], [2, 4, 6, 8]], ids=lambda x: f"{len(x)}D"
)
def test_batch_norm_conv_fusing(bias: bool, input_shape: list[int]):
    example_input = (torch.ones(*input_shape),)

    module = ConvBatchNormModule(bias, len(input_shape), 4)
    program = torch.export.export(module, example_input, strict=True)
    og_module = program.module()

    pm = NeutronAtenPassManager(neutron_target_spec)
    graph_module_out = pm(deepcopy(program.module())).graph_module

    # Make sure the fusion worked.
    og_nodes = list(program.graph.nodes)
    transformed_nodes = list(graph_module_out.graph.nodes)

    assert any(
        node.op == "call_function" and node.target.__name__ == "batch_norm.default"
        for node in og_nodes
    )

    assert not any(
        node.op == "call_function" and "batch_norm" in node.target.__name__
        for node in transformed_nodes
    )

    # Verify that the behavior has not changed.
    input_data = torch.randn(input_shape, dtype=torch.float32)
    out1 = og_module(input_data).detach().numpy()
    out2 = graph_module_out(input_data).detach().numpy()
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_linear_fusing(bias: bool):
    input_shape = (2, 4, 6, 8)
    example_input = (torch.ones(*input_shape),)

    module = LinearBatchNormModule(bias, 4, input_shape[-1], input_shape[1])
    program = torch.export.export(module, example_input, strict=True)
    og_module = program.module()

    pm = NeutronAtenPassManager(neutron_target_spec)
    graph_module_out = pm(deepcopy(program.module())).graph_module

    # Make sure the fusion worked.
    og_nodes = list(og_module.graph.nodes)
    transformed_nodes = list(graph_module_out.graph.nodes)

    assert any(
        node.op == "call_function" and node.target.__name__ == "linear.default"
        for node in og_nodes
    )

    assert not any(
        node.op == "call_function" and "batch_norm" in node.target.__name__
        for node in transformed_nodes
    )

    # Verify that the behavior has not changed.
    input_data = torch.randn(input_shape, dtype=torch.float32)
    out1 = og_module(input_data).detach().numpy()
    out2 = graph_module_out(input_data).detach().numpy()
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_conv_fusing__full_pipeline__1d(bias: bool):
    input_shape = [4, 6, 8]
    module = ConvBatchNormModule(bias, len(input_shape), 4)

    edge_program = to_quantized_edge_program(
        module, tuple(input_shape)
    ).exported_program()
    nodes = list(edge_program.graph.nodes)

    assert (
        len(nodes) == 17
    )  # 1D Conv currently isn't delegated, because it doesn't get quantized.
    assert not any(
        node.op == "call_function" and "batch_norm" in node.target.__name__
        for node in nodes
    )


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_conv_fusing__full_pipeline__2d(bias: bool):
    input_shape = [1, 4, 6, 8]
    module = ConvBatchNormModule(bias, len(input_shape), 4)

    edge_program = to_quantized_edge_program(
        module, tuple(input_shape)
    ).exported_program()
    nodes = list(edge_program.graph.nodes)

    assert len(nodes) == 7
    assert not any(
        node.op == "call_function" and "batch_norm" in node.target.__name__
        for node in nodes
    )


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_linear_fusing__full_pipeline(bias: bool):
    input_shape = (2, 4, 6, 8)
    module = LinearBatchNormModule(bias, 4, input_shape[-1], input_shape[1])

    # Don't delegate the Linear node, because there seems to be a bug with the NeutronConverter/NeutronPartitioner.
    #  But that doesn't affect the validity of this test.
    def unsupported_target(*_):  # Accept all input arguments and return `False`.
        return False

    with OverrideTargetSupportCheck(
        AddMMConverter, new_target_support_check=unsupported_target
    ):
        with OverrideTargetSupportCheck(
            MMConverter, new_target_support_check=unsupported_target
        ):
            with OverrideTargetSupportCheck(
                ViewCopyConverter, new_target_support_check=unsupported_target
            ):
                edge_program = to_quantized_edge_program(
                    module, tuple(input_shape)
                ).exported_program()
                nodes = list(edge_program.graph.nodes)

    assert len(nodes) == 18
    assert not any(
        node.op == "call_function" and "batch_norm" in node.target.__name__
        for node in nodes
    )


@pytest.mark.parametrize(
    "conv_module",
    ["conv2d", "conv2d_t"],
)
def test_biasless_convbn_fusion_qat(
    mocker,
    conv_module,
):
    if conv_module.startswith("conv1d"):
        input_shape = (1, 3, 32)
    elif conv_module.startswith("conv2d"):
        input_shape = (1, 3, 32, 32)
    else:  # conv3d
        input_shape = (1, 3, 32, 32, 32)

    model = ConvBNModule(conv_module, conv_bias=False, bn_affine=True)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=True, use_neutron_for_format_conversion=False
    ).exported_program()

    assert any("lowered_module" in node.name for node in edge_program.graph.nodes)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
        atol=1.0,
    )
