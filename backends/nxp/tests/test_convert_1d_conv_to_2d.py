# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    ConvertConv1dToConv2dPass,
    NeutronAtenPassManager,
)

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    neutron_target_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
)
from executorch.backends.nxp.tests.models import Conv1dModule, ConvTranspose1dModule
from executorch.exir.dialects._ops import ops as exir_ops
from torch import nn
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


AtenConv1d = torch.ops.aten.conv1d.default
AtenConv2d = torch.ops.aten.conv2d.default
AtenConvTranspose1d = torch.ops.aten.conv_transpose1d.default
AtenConvTranspose2d = torch.ops.aten.conv_transpose2d.input
AtenSqueeze = torch.ops.aten.squeeze.dim
AtenUnsqueeze = torch.ops.aten.unsqueeze.default
AtenRelu = torch.ops.aten.relu.default
AtenSigmoid = torch.ops.aten.sigmoid.default
AtenTanh = torch.ops.aten.tanh.default
AtenHardtanh = torch.ops.aten.hardtanh.default
AtenBatchNorm = torch.ops.aten.batch_norm.default

EdgeConvolution = exir_ops.edge.aten.convolution.default
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation, groups, bias",
    [
        pytest.param((3, 7, 23), 3, 1, 0, 1, 1, True, id="All default."),
        pytest.param(
            (3, 7), 3, 1, 0, 1, 1, True, id="All default, implicit `batch` dim."
        ),
        pytest.param(
            (3, 7, 23), 2, 1, 0, 1, 1, True, id="kernel_size=2, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23), 3, 2, 0, 1, 1, True, id="stride=2, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23), 3, 1, 1, 1, 1, True, id="pad=1, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23), 3, 1, 0, 2, 1, True, id="dilation=2, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23), 3, 1, 0, 1, 7, True, id="group=7, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23), 3, 1, 0, 1, 1, False, id="bias=False, otherwise all default."
        ),
        pytest.param((3, 7, 23), 5, 3, 2, 3, 7, False, id="Nothing is default."),
    ],
)
def test_convert_conv_1d_to_conv2d(
    input_shape, kernel_size, stride, padding, dilation, groups, bias
):
    if len(input_shape) == 2:
        in_channels = input_shape[0]
    else:
        in_channels = input_shape[1]
    out_channels = 14
    model = Conv1dModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )
    example_input = torch.rand(input_shape)

    exir_program_aten = torch.export.export(model, (example_input,)).module()

    # Make sure `aten.conv1d` is present.
    assert graph_contains_any_of_ops(exir_program_aten.graph, [AtenConv1d])
    outputs_before = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Apply the optimization.
    NeutronAtenPassManager(
        neutron_target_spec, [ConvertConv1dToConv2dPass(neutron_target_spec)]
    )(exir_program_aten)

    # Make sure no `aten.conv1d` nodes are in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph,
        [
            AtenConv1d,
        ],
    )

    # Check correct count and placement.
    nodes = list(exir_program_aten.graph.nodes)

    conv_nodes = [i for i, n in enumerate(nodes) if n.target == AtenConv2d]
    assert len(conv_nodes) == 1
    i = conv_nodes[0]

    assert nodes[i - 1].target == AtenUnsqueeze
    assert nodes[i].target == AtenConv2d
    assert nodes[i + 1].target == AtenSqueeze

    outputs_after = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Make sure the model still produces the exact same output.
    assert len(outputs_before) == len(outputs_after)
    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i])


# Note: The first case is the default; the remaining cases are chosen to test various parameter combinations.
# To satisfy requirements for delegation, some parameters could not be chosen arbitrarily.
@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, output_padding, groups, bias, dilation",
    [
        pytest.param((3, 7, 23), 3, 1, 0, 0, 1, True, 1, id="All default."),
        pytest.param(
            (3, 7), 3, 1, 0, 0, 1, True, 1, id="All default, implicit `batch` dim."
        ),
        pytest.param(
            (3, 7, 23),
            2,
            1,
            0,
            0,
            1,
            True,
            1,
            id="kernel_size=2, otherwise all default.",
        ),
        pytest.param(
            (3, 7, 23), 3, 2, 0, 0, 1, True, 1, id="stride=2, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23), 3, 1, 1, 0, 1, True, 1, id="pad=1, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23),
            3,
            2,
            0,
            1,
            1,
            True,
            1,
            id="output_padding=1 (stride=2 - restriction from definition), otherwise all default.",
        ),
        pytest.param(
            (3, 7, 23), 3, 1, 0, 0, 7, True, 1, id="group=7, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23), 3, 1, 0, 0, 1, False, 1, id="bias=False, otherwise all default."
        ),
        pytest.param(
            (3, 7, 23), 3, 1, 0, 0, 1, True, 2, id="dilation=2, otherwise all default."
        ),
        pytest.param((3, 7, 23), 5, 3, 2, 1, 7, False, 3, id="Nothing is default."),
    ],
)
def test_convert_conv_1d_transp_to_conv2d_transp(
    input_shape, kernel_size, stride, padding, output_padding, groups, bias, dilation
):
    if len(input_shape) == 2:
        in_channels = input_shape[0]
    else:
        in_channels = input_shape[1]
    out_channels = 14
    model = ConvTranspose1dModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )
    example_input = torch.rand(input_shape)

    exir_program_aten = torch.export.export(model, (example_input,)).module()

    # Make sure `aten.conv_transpose1d` is present.
    assert graph_contains_any_of_ops(exir_program_aten.graph, [AtenConvTranspose1d])
    outputs_before = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Apply the optimization.
    NeutronAtenPassManager(
        neutron_target_spec, [ConvertConv1dToConv2dPass(neutron_target_spec)]
    )(exir_program_aten)

    # Make sure no `aten.conv_transpose1d` nodes are in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph,
        [
            AtenConvTranspose1d,
        ],
    )

    # Check correct count and placement.
    nodes = list(exir_program_aten.graph.nodes)

    conv_nodes = [i for i, n in enumerate(nodes) if n.target == AtenConvTranspose2d]
    assert len(conv_nodes) == 1
    i = conv_nodes[0]

    assert nodes[i - 1].target == AtenUnsqueeze
    assert nodes[i].target == AtenConvTranspose2d
    assert nodes[i + 1].target == AtenSqueeze

    outputs_after = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Make sure the model still produces the exact same output.
    assert len(outputs_before) == len(outputs_after)
    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i])


# Note: The first case is the default; the remaining cases are chosen to test various parameter combinations.
# To satisfy requirements for delegation, some parameters could not be chosen arbitrarily.
@pytest.mark.parametrize("input_shape", [(1, 8, 24), (8, 24)])
@pytest.mark.parametrize("use_qat", [True, False])
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, groups, bias",
    [
        pytest.param(3, 1, 1, 1, 1, True, id="All default, except for padding = 1."),
        pytest.param(1, 1, 0, 1, 1, True, id="kernel_size = 1"),
        pytest.param(3, 2, 5, 1, 1, True, id="stride = 2"),
        pytest.param(3, 1, 2, 2, 1, True, id="dilation = 2"),
        pytest.param(3, 1, 1, 1, 1, False, id="bias = False, padding = 1"),
    ],
)
def test_convert_conv_1d_to_conv2d_full_pipeline(
    mocker, input_shape, kernel_size, stride, padding, dilation, groups, bias, use_qat
):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    in_channels = input_shape[1] if len(input_shape) == 3 else input_shape[0]
    out_channels = 16

    model = Conv1dModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure no `conv1d` nodes are in the model.
    assert not graph_contains_any_of_ops(
        delegated_ep.graph,
        [
            AtenConv1d,
        ],
    )

    # Check correct count and placement.
    nodes = list(delegated_ep.graph.nodes)
    assert len(nodes) == 7
    assert nodes[3].target == ExecutorchDelegateCall

    # Capture generated model.
    neutron_ir_model = converter_spy.spy_return[0]
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    # Make sure `edge.aten.convolution.default` is in the model.
    assert graph_contains_any_of_ops(
        exported_program.graph,
        [EdgeConvolution],
    )

    example_input = (np.random.random(input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    convert_run_compare(
        exported_program,
        input_data=example_input,
        tfl_model=neutron_ir_model,
    )


# Note: The first case is the default; the remaining cases are chosen to test various parameter combinations.
# To satisfy requirements for delegation, some parameters could not be chosen arbitrarily.
@pytest.mark.parametrize("input_shape", [(1, 8, 24), (8, 24)])
@pytest.mark.parametrize("use_qat", [False, True])
@pytest.mark.parametrize(
    "kernel_size, stride, padding, output_padding, groups, bias, dilation",
    [
        pytest.param(2, 2, 0, 0, 1, True, 1, id="All default."),
        pytest.param(4, 2, 1, 0, 1, True, 1, id="kernel_size = 4 (and padding = 1)"),
        pytest.param(4, 4, 0, 0, 1, True, 1, id="stride = 4 (and kernel_size = 4)"),
        pytest.param(
            4,
            4,
            1,
            2,
            1,
            True,
            1,
            id="output_padding = 2 (and kernel_size = 4, stride = 4, padding = 1)",
            marks=pytest.mark.skip(reason="Neutron Converter hangs (AIR-14771)."),
        ),
        pytest.param(2, 2, 0, 0, 1, False, 1, id="bias=False"),
    ],
)
def test_convert_conv_1d_to_conv2d_transp_full_pipeline(
    mocker,
    input_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    groups,
    bias,
    dilation,
    use_qat,
):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    in_channels = input_shape[1] if len(input_shape) == 3 else input_shape[0]
    out_channels = 16
    model = ConvTranspose1dModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )

    # Run conversion.
    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure no `aten.conv_transpose1d` nodes are in the model.
    assert not graph_contains_any_of_ops(
        delegated_ep.graph,
        [AtenConvTranspose1d],
    )

    # Check correct count and placement.
    nodes = list(delegated_ep.graph.nodes)
    assert len(nodes) == 7
    assert nodes[3].target == ExecutorchDelegateCall

    # Capture generated model.
    neutron_ir_model = converter_spy.spy_return[0]
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    # Make sure `edge.aten.convolution.default` is in the model.
    assert graph_contains_any_of_ops(
        exported_program.graph,
        [EdgeConvolution],
    )

    example_input = (np.random.random(input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    convert_run_compare(
        exported_program,
        input_data=example_input,
        tfl_model=neutron_ir_model,
    )


class Conv1dActivationModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, activation, stride=1, padding=0
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))


class Conv1dBNActivationModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, activation, stride=1, padding=0
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class Conv1dHardtanhUnsupportedModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)

    def forward(self, x):
        return self.hardtanh(self.conv(x))


@pytest.mark.parametrize(
    "activation, expected_act_target, has_bn",
    [
        pytest.param(nn.ReLU(), AtenRelu, False, id="conv1d_relu"),
        pytest.param(nn.ReLU(), AtenRelu, True, id="conv1d_bn_relu"),
        pytest.param(nn.Sigmoid(), AtenSigmoid, False, id="conv1d_sigmoid"),
        pytest.param(nn.Sigmoid(), AtenSigmoid, True, id="conv1d_bn_sigmoid"),
        pytest.param(nn.Tanh(), AtenTanh, False, id="conv1d_tanh"),
        pytest.param(nn.Tanh(), AtenTanh, True, id="conv1d_bn_tanh"),
        pytest.param(
            nn.Hardtanh(min_val=0.0, max_val=6.0),
            AtenHardtanh,
            False,
            id="conv1d_relu6",
        ),
        pytest.param(
            nn.Hardtanh(min_val=0.0, max_val=6.0),
            AtenHardtanh,
            True,
            id="conv1d_bn_relu6",
        ),
    ],
)
def test_convert_conv_1d_to_conv2d_keeps_activation_in_4d(
    activation, expected_act_target, has_bn
):
    input_shape = (3, 7, 23)
    model_cls = Conv1dBNActivationModule if has_bn else Conv1dActivationModule
    model = model_cls(
        in_channels=7, out_channels=14, kernel_size=3, activation=activation, padding=1
    )
    example_input = torch.rand(input_shape)

    exir_program_aten = torch.export.export(model, (example_input,)).module()

    assert graph_contains_any_of_ops(exir_program_aten.graph, [AtenConv1d])
    outputs_before = [o.detach().numpy() for o in exir_program_aten(example_input)]

    NeutronAtenPassManager(
        neutron_target_spec, [ConvertConv1dToConv2dPass(neutron_target_spec)]
    )(exir_program_aten)

    assert not graph_contains_any_of_ops(exir_program_aten.graph, [AtenConv1d])

    nodes = list(exir_program_aten.graph.nodes)
    conv_nodes = [i for i, n in enumerate(nodes) if n.target == AtenConv2d]
    assert len(conv_nodes) == 1
    i = conv_nodes[0]

    assert nodes[i - 1].target == AtenUnsqueeze
    assert nodes[i].target == AtenConv2d

    if has_bn:
        assert nodes[i + 1].target == AtenBatchNorm
        assert nodes[i + 2].target == expected_act_target
        assert nodes[i + 3].target == AtenSqueeze
    else:
        assert nodes[i + 1].target == expected_act_target
        assert nodes[i + 2].target == AtenSqueeze

    outputs_after = [o.detach().numpy() for o in exir_program_aten(example_input)]

    assert len(outputs_before) == len(outputs_after)
    for j in range(len(outputs_before)):
        assert np.allclose(outputs_before[j], outputs_after[j])


def test_convert_conv_1d_to_conv2d_unsupported_hardtanh_not_fused():
    input_shape = (3, 7, 23)
    model = Conv1dHardtanhUnsupportedModule(
        in_channels=7, out_channels=14, kernel_size=3, padding=1
    )
    example_input = torch.rand(input_shape)

    exir_program_aten = torch.export.export(model, (example_input,)).module()

    assert graph_contains_any_of_ops(exir_program_aten.graph, [AtenConv1d])
    outputs_before = [o.detach().numpy() for o in exir_program_aten(example_input)]

    NeutronAtenPassManager(
        neutron_target_spec, [ConvertConv1dToConv2dPass(neutron_target_spec)]
    )(exir_program_aten)

    assert not graph_contains_any_of_ops(exir_program_aten.graph, [AtenConv1d])

    nodes = list(exir_program_aten.graph.nodes)
    conv_nodes = [i for i, n in enumerate(nodes) if n.target == AtenConv2d]
    assert len(conv_nodes) == 1
    i = conv_nodes[0]

    assert nodes[i - 1].target == AtenUnsqueeze
    assert nodes[i].target == AtenConv2d
    assert nodes[i + 1].target == AtenSqueeze
    assert nodes[i + 2].target == AtenHardtanh

    outputs_after = [o.detach().numpy() for o in exir_program_aten(example_input)]

    assert len(outputs_before) == len(outputs_after)
    for j in range(len(outputs_before)):
        assert np.allclose(outputs_before[j], outputs_after[j])
