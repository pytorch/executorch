# Copyright 2024-2025 NXP
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
        Conv2dModule(bias=False),
        (1, 4, 32, 32),
        use_neutron_for_format_conversion=False,
    )
    payload = (
        edge_program_manager.exported_program().graph_module.lowered_module_0.processed_bytes
    )

    assert payload[0] == 0x1  # Number of Neutron node inputs
    assert payload[1] == 0x1  # Number of Neutron node outputs
    assert payload[2] == 0x1  # Number of model inputs
    assert payload[3] == 0x1  # Channels last 0-th Neutron input
    assert payload[4] == 0x1  # Channels last 0-th Neutron output
    assert payload[5] == 0x0  # Map 0-th Neutron input to 0-th model input
    assert payload[6] == 0x0  # Map 0-th Neutron output to 0-th model output
    assert (
        payload[7] == 0x0 or payload[7] == 0x1
    )  # Payload version is 0 or 1 depending on the Neutron Software
    assert all(byte == 0x0 for byte in payload[8:16])  # Aligned to 16 bytes
    assert payload[17] != 0x0  # Followed by non-zero content


def test_neutron_backend__linear_softmax_model__payload_header_formatless():
    edge_program_manager = to_quantized_edge_program(LinearSoftmaxModule(), (1, 12))
    payload = (
        edge_program_manager.exported_program().graph_module.lowered_module_0.processed_bytes
    )

    assert payload[0] == 0x1  # Number of Neutron node inputs
    assert payload[1] == 0x1  # Number of Neutron node outputs
    assert payload[2] == 0x1  # Number of model inputs
    assert payload[3] == 0x0  # Formatless 0-th Neutron input
    assert payload[4] == 0x0  # Formatless 0-th Neutron output
    assert payload[5] == 0x0  # Map 0-th Neutron input to 0-th model input
    assert payload[6] == 0x0  # Map 0-th Neutron output to 0-th model output
    assert (
        payload[7] == 0x0 or payload[7] == 0x1
    )  # Payload version is 0 or 1 depending on the Neutron Software
    assert all(byte == 0x0 for byte in payload[8:16])  # Aligned to 16 bytes
    assert payload[17] != 0x0  # Followed by non-zero content
