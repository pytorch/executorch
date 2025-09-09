# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import executorch.extension.pybindings.portable_lib
import executorch.kernels.quantized  # noqa F401
import torch
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.models import Conv2dReLUModule
from executorch.examples.nxp.experimental.cifar_net.cifar_net import CifarNet
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.quantize_io_pass import get_config_method_name


def test_remove_io_quant_ops_pass__conv_relu():
    model = Conv2dReLUModule()
    model.eval()

    input_shape = (1, 4, 32, 32)
    edge_program_manager = to_quantized_edge_program(
        model, input_shape, remove_quant_io_ops=True
    )

    exec_prog = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    nodes = list(exec_prog.exported_program().graph.nodes)
    assert (
        nodes[0].meta["val"].dtype == torch.int8
    ), "Input tensor doesn't have type INT8."
    assert nodes[2].name == "executorch_call_delegate"
    assert (
        nodes[4].meta["val"][0].dtype == torch.int8
    ), "Output tensor doesn't have type INT8."

    assert (
        get_config_method_name(None, "input", 0, "scale") in exec_prog._config_methods
    )
    assert get_config_method_name(None, "input", 0, "zp") in exec_prog._config_methods
    assert (
        get_config_method_name(None, "output", 0, "scale") in exec_prog._config_methods
    )
    assert get_config_method_name(None, "output", 0, "zp") in exec_prog._config_methods


def test_remove_io_quant_ops_pass__cifarnet():
    model = CifarNet().get_eager_model()
    input_shape = (1, 3, 32, 32)
    edge_program_manager = to_quantized_edge_program(
        model, input_shape, remove_quant_io_ops=True
    )

    exec_prog = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    nodes = list(exec_prog.exported_program().graph.nodes)
    assert len(nodes) == 9
    assert (
        nodes[0].meta["val"].dtype == torch.int8
    ), "Input tensor doesn't have type INT8."
    # Currently, softmax is not quantized
    assert (
        nodes[8].meta["val"][0].dtype == torch.float32
    ), "Output tensor doesn't have type INT8."

    assert (
        get_config_method_name(None, "input", 0, "scale") in exec_prog._config_methods
    )
    assert get_config_method_name(None, "input", 0, "zp") in exec_prog._config_methods


class MultiInputOutputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(4, 64, 2, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x, y):
        z = self.relu(x)
        x = self.conv(z)
        return x + y, z


def test_multiple_inputs__multiple_outputs():
    model = MultiInputOutputModule()
    model.eval()

    input_shape = [(1, 4, 32, 32), (1, 1, 1, 31)]
    edge_program_manager = to_quantized_edge_program(
        model, input_shape, remove_quant_io_ops=True
    )

    exec_prog = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    nodes = list(exec_prog.exported_program().graph.nodes)
    print(nodes)
    assert (
        nodes[0].meta["val"].dtype == torch.int8
    ), "Input tensor doesn't have type INT8."
    assert nodes[3].name == "executorch_call_delegate"
    assert (
        nodes[-1].meta["val"][0].dtype == torch.int8
    ), "Output tensor doesn't have type INT8."

    quant_method_variants = itertools.product(
        ["input", "output"], [0, 1], ["scale", "zp"]
    )

    expected_methods = [
        get_config_method_name(None, arg_type, index, key)
        for arg_type, index, key in quant_method_variants
    ]
    assert all(method in exec_prog._config_methods for method in expected_methods)
