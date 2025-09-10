# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.extension.pybindings.portable_lib
import executorch.kernels.quantized  # noqa F401

from executorch.backends.nxp.tests.executorch_pipeline import (
    to_quantized_executorch_program,
)
from executorch.backends.nxp.tests.models import ConvFCSoftmaxModule
from executorch.devtools.backend_debug import get_delegation_info
from executorch.examples.nxp.experimental.cifar_net.cifar_net import CifarNet


def test_conv_fc_softmax__to_executorch_program():
    model = ConvFCSoftmaxModule()
    input_shape = (1, 4, 5, 5)

    exec_prog = to_quantized_executorch_program(model, input_shape)

    program = exec_prog.exported_program()
    assert (
        program.graph_module.lowered_module_0
    ), "There is no lowered module with Neutron microcode."

    delegation_info = get_delegation_info(program.graph_module)
    assert delegation_info.num_delegated_subgraphs == 1
    assert delegation_info.num_non_delegated_nodes == 7
    assert delegation_info.num_delegated_nodes == 13

    for node in program.graph.nodes:
        # Make sure Convolution and AddMM are delegated
        assert "convolution" not in node.name
        assert "addmm" not in node.name


def test_cifarnet():
    model = CifarNet().get_eager_model().eval()
    input_shape = (1, 3, 32, 32)
    exec_prog = to_quantized_executorch_program(model, input_shape)

    delegation_info = get_delegation_info(exec_prog.exported_program().graph_module)
    assert delegation_info.num_delegated_subgraphs == 1
    assert delegation_info.num_non_delegated_nodes == 7
    assert delegation_info.num_delegated_nodes == 45

    nodes = list(exec_prog.exported_program().graph.nodes)
    assert nodes[2].name == "quantized_decomposed_quantize_per_tensor_default"
