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
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.tests.executorch_pipeline import to_edge_program
from executorch.backends.nxp.tests.executors import convert_run_compare
from executorch.backends.nxp.tests.models import SoftmaxConvModule, SoftmaxModule


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        pytest.param((10,), -1, id="1D,dim=-1"),
        pytest.param((10,), 0, id="1D,dim=0"),
        pytest.param((10, 32), -1, id="2D,dim=-1"),
        pytest.param((10, 32), 1, id="2D,dim=1"),
    ],
)
def test_softmax_conversion__formatless_input(input_shape, dim: int):
    model = SoftmaxModule(dim)

    edge_program = to_edge_program(model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(edge_program, input_data=input_data)


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        pytest.param((10, 32, 32), -1, id="3D,dim=-1"),
        pytest.param((10, 32, 32), 2, id="3D,dim=2"),
        pytest.param((10, 32, 32, 8), -1, id="4D,dim=-1"),
        pytest.param((10, 32, 32, 8), 3, id="4D,dim=3"),
        pytest.param((10, 32, 32, 8, 8), -1, id="5D,dim=-1"),
        pytest.param((10, 32, 32, 8, 8), 4, id="5D,dim=4"),
    ],
)
def test_softmax_conversion__unknown_input_format(input_shape, dim: int):
    model = SoftmaxModule(dim)

    edge_program = to_edge_program(model, input_shape).exported_program()

    # Currently this test not pass because the convertibility checker doesn't use tensor formats.
    with pytest.raises(
        AssertionError, match="`aten__softmax_default` is not convertible"
    ):
        EdgeProgramToIRConverter().convert_program(edge_program, ConversionConfig())

    # input_data = np.random.random(input_shape).astype(np.float32)
    # convert_run_compare(edge_program_manager.exported_program(), input_data=input_data, atol=5e-7)


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        pytest.param((1, 4, 32, 32), 1, id="4D,dim=1"),
        pytest.param((1, 4, 16, 16), -3, id="4D,dim=-3"),
    ],
)
def test_softmax_conversion_channel_last(input_shape, dim: int):
    model = SoftmaxConvModule(dim)

    edge_program = to_edge_program(model, input_shape).exported_program()

    # TODO (Robert Kalmar) Currently this test not pass because the convertibility checker doesn't use tensor formats.
    with pytest.raises(
        AssertionError, match="`aten__softmax_default` is not convertible"
    ):
        EdgeProgramToIRConverter().convert_program(edge_program, ConversionConfig())

    # input_data = np.random.random(input_shape).astype(np.float32)
    # convert_run_compare(edge_program_manager.exported_program(), tflite_input_preprocess=ToNHWCPreprocess(),
    #                     tflite_output_preprocess=ToNCHWPreprocess(), input_data=input_data, atol=5e-7)


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        pytest.param((10, 32), 0, id="2D,dim=0"),
        pytest.param((10, 32, 32), 1, id="3D,dim=1"),
        pytest.param((10, 32, 32, 8), 2, id="4D,dim=2"),
        pytest.param((10, 32, 32, 8, 8), 3, id="5D,dim=3"),
        pytest.param((10, 32, 32, 8, 8), 2, id="5D,dim=2"),
    ],
)
def test_softmax_conversion_unsupported_dims(input_shape, dim: int):
    model = SoftmaxModule(dim)

    edge_program = to_edge_program(model, input_shape).exported_program()

    with pytest.raises(
        AssertionError, match="`aten__softmax_default` is not convertible"
    ):
        EdgeProgramToIRConverter().convert_program(edge_program, ConversionConfig())
