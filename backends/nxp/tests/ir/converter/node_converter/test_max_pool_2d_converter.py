# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.neutron_pass_manager import NeutronPassManager
from executorch.backends.nxp.tests.executorch_pipeline import (
    to_edge_program,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.backends.nxp.tests.models import MaxPool2dConvModule, MaxPool2dModule
from executorch.backends.xnnpack._passes import RemoveGetItemPass
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((1, 4, 8, 8), (0, 0), id="No padding."),
        pytest.param(
            (1, 4, 8, 8),
            (1, 1),
            id="Padding, keep the same output tensor size as input.",
        ),
        pytest.param(
            (1, 4, 8, 8), (1, 0), id="Padding, change the output tensor size."
        ),
        pytest.param(
            (1, 4, 9, 9), (1, 0), id="Padding, change the output tensor size."
        ),
        pytest.param(
            (1, 4, 9, 9), (0, 1), id="Padding, change the output tensor size."
        ),
    ],
)
def test_max_pool_2d_conversion(input_shape, padding):
    edge_program = to_edge_program(
        MaxPool2dModule(padding=padding), input_shape
    ).exported_program()

    # We need to create custom model verifier with max_pool2d added as exception.
    # Otherwise, we get violation that this op is not part of ATen Core ops.
    edge_program._verifiers = [
        EXIREdgeDialectVerifier(
            class_only=True,
            core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
        )
    ]

    # Remove MaxPool-related "getitem" nodes from graph
    edge_program = NeutronPassManager(edge_program, [RemoveGetItemPass]).transform()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
    )


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((1, 4, 8, 8), (0, 0), id="No padding."),
        pytest.param(
            (1, 4, 8, 8),
            (1, 1),
            id="Padding, keep the same output tensor size as input.",
        ),
        pytest.param(
            (1, 4, 8, 8), (1, 0), id="Padding, change the output tensor size."
        ),
        pytest.param(
            (1, 4, 11, 11), (1, 0), id="Padding, change the output tensor size."
        ),
        pytest.param(
            (1, 4, 11, 11), (0, 1), id="Padding, change the output tensor size."
        ),
    ],
)
def test_max_pool_2d_quant_conversion(mocker, input_shape, padding):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(MaxPool2dConvModule(padding=padding), input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
    )
