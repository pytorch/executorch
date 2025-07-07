# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.models import Conv2dModule, LinearSoftmaxModule


def test_neutron_backend__single_conv_model():
    edge_program_manager = to_quantized_edge_program(
        Conv2dModule(bias=False), (1, 4, 32, 32)
    )
    lowered_module = (
        edge_program_manager.exported_program().graph_module.lowered_module_0
    )
    assert (
        len(lowered_module.processed_bytes) != 0
    )  # The Neutron microcode, weights and kernels have been written here


def test_neutron_backend__single_conv_model__payload_header_channels_last():
    edge_program_manager = to_quantized_edge_program(
        Conv2dModule(bias=False), (1, 4, 32, 32)
    )
    payload = (
        edge_program_manager.exported_program().graph_module.lowered_module_0.processed_bytes
    )

    assert payload[0] == 0x1  # Single input
    assert payload[1] == 0x1  # Single output
    assert payload[2] == 0x1  # Channels last
    assert payload[3] == 0x1  # Channels last
    assert all(byte == 0x0 for byte in payload[4:16])  # Aligned to 16 bytes
    assert payload[17] != 0x0  # Followed by non-zero content


def test_neutron_backend__linear_softmax_model__payload_header_formatless():
    edge_program_manager = to_quantized_edge_program(LinearSoftmaxModule(), (1, 12))
    payload = (
        edge_program_manager.exported_program().graph_module.lowered_module_0.processed_bytes
    )

    assert payload[0] == 0x1  # Single input
    assert payload[1] == 0x1  # Single output
    assert payload[2] == 0x0  # Formatless
    assert payload[3] == 0x0  # Formatless
    assert all(byte == 0x0 for byte in payload[4:16])  # Aligned to 16 bytes
    assert payload[17] != 0x0  # Followed by non-zero content
