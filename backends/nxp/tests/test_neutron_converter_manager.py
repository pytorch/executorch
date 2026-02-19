# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch import exir
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.neutron_converter_manager import (
    NeutronConverterManager,
)
from executorch.backends.nxp.backend.node_format_inference import NodeFormatInference
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.models import Conv2dModule, LinearModule


def test_conv2d_neutron_conversion__default_flavor():
    model = Conv2dModule()

    example_input = (torch.ones(1, 4, 32, 32),)
    exir_program = torch.export.export(model, example_input)
    edge_program_manager = exir.to_edge(exir_program)

    NodeFormatInference(edge_program_manager.exported_program()).identify_node_formats()
    edge_program_converter = EdgeProgramToIRConverter()
    tflite_model, _ = edge_program_converter.convert_program(
        edge_program_manager.exported_program()
    )

    neutron_converter_manager = NeutronConverterManager()
    neutron_model = neutron_converter_manager.convert(tflite_model, "imxrt700", False)

    assert len(
        neutron_model
    ), "Produced NeutronGraph-based TFLite model has zero length!"


def test__conv2d_neutron_conversion__invalid_flavor():
    model = Conv2dModule()

    example_input = (torch.ones(1, 4, 32, 32),)
    exir_program = torch.export.export(model, example_input)
    edge_program_manager = exir.to_edge(exir_program)

    NodeFormatInference(edge_program_manager.exported_program()).identify_node_formats()
    edge_program_converter = EdgeProgramToIRConverter()
    tflite_model, _ = edge_program_converter.convert_program(
        edge_program_manager.exported_program()
    )

    with pytest.raises(RuntimeError) as excinfo:
        _ = NeutronConverterManager("bad_flavor").convert(
            tflite_model, "imxrt700", False
        )

    assert "Neutron Converter module with flavor 'bad_flavor' not found." in str(
        excinfo
    )


def test_conv2d_neutron_conversion__prefetching(mocker):
    model = LinearModule(True)
    input_shape = (1, 1, 32, 32)

    converter_spy = mocker.spy(NeutronConverterManager, "convert")
    _ = to_quantized_edge_program(
        model, input_shape, fetch_constants_to_sram=True
    ).exported_program()
    neutron_model_prefetch = converter_spy.spy_return

    _ = to_quantized_edge_program(
        model, input_shape, fetch_constants_to_sram=False
    ).exported_program()
    neutron_model_regular = converter_spy.spy_return

    assert len(neutron_model_prefetch) != len(
        neutron_model_regular
    ), "The weight prefetching flag does not make a difference!"
