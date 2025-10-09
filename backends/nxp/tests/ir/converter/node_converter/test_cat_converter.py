# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


def _normalized_dim(dim, rank):
    return dim if dim >= 0 else dim + rank


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class CatModule(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs: torch.Tensor):
        return torch.cat(list(inputs), self.dim)


class AddCatModule(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs: torch.Tensor):
        inputs = [input_ + input_ for input_ in inputs]

        return torch.cat(list(inputs), self.dim)


class CatConvModule(torch.nn.Module):

    def __init__(self, dim: int, channels: int = 4):
        super().__init__()
        self.dim = dim
        self.conv = torch.nn.Conv2d(channels, channels, 2)

    def forward(self, *inputs: torch.Tensor):
        x = torch.cat(list(inputs), self.dim)
        return self.conv(x)


@pytest.mark.parametrize(
    "rank, num_inputs, dim",
    [
        pytest.param(2, 2, 1, id="2D, 2 inputs, dim=1"),
        pytest.param(2, 2, -1, id="2D, 2 inputs, dim=-1"),
        pytest.param(2, 3, 1, id="2D, 3 inputs, dim=1"),
        pytest.param(2, 3, -1, id="2D, 3 inputs, dim=-1"),
        pytest.param(2, 4, -1, id="2D, 4 inputs, dim=-1"),
        pytest.param(3, 2, 1, id="3D, 2 inputs, dim=1"),
        pytest.param(3, 2, -1, id="3D, 2 inputs, dim=-1"),
        pytest.param(3, 5, -1, id="3D, 5 inputs, dim=-2"),
        pytest.param(4, 2, -1, id="4D, 2 inputs, dim=-1"),
        pytest.param(4, 3, 2, id="4D, 3 inputs, dim=2"),
        pytest.param(4, 5, -3, id="4D, 5 inputs, dim=-3"),
    ],
)
def test_cat__same_shapes(dim, num_inputs, rank, mocker):
    input_shape = tuple([8, 8, 8, 8][:rank])

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(
        CatModule(dim), [input_shape] * num_inputs
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    input_data = {
        i: (np.random.random(input_shape) * 50).astype(np.int8)
        for i in range(num_inputs)
    }
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        input_data=input_data,
        atol=1,
    )


@pytest.mark.parametrize("dim", [3, -2, -3])
@pytest.mark.parametrize("num_inputs", [2, 5])
def test_cat__channels_first__same_shapes(dim, num_inputs, mocker):
    input_shape = (2, 8, 6, 8)
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    channels = input_shape[1] if dim not in {1, -3} else input_shape[1] * num_inputs
    quantized_program = to_quantized_edge_program(
        CatConvModule(dim, channels), [input_shape] * num_inputs
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    input_data = {
        i: (np.random.random(input_shape) * 50).astype(np.int8)
        for i in range(num_inputs)
    }
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        input_data=input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        atol=1,
    )


@pytest.mark.parametrize(
    "dim, input_shape",
    [
        pytest.param(0, (1, 8, 8, 8), id="axis = 0"),
        pytest.param(0, (8, 8, 8, 8), id="axis = 0, no `1s` in the shape."),
        pytest.param(-4, (1, 8, 8, 8), id="axis = -4"),
        pytest.param(1, (1, 1, 8, 8), id="axis = 1"),
        pytest.param(-3, (1, 1, 8, 8), id="axis = -3"),
        pytest.param(2, (1, 1, 1, 8), id="axis = 2"),
        pytest.param(-2, (1, 1, 1, 8), id="axis = -2"),
    ],
)
def test_cat__unsupported__imxrt700(dim, input_shape):
    """This test is conjoined with the one below (`test_cat__context_dependent__imxrt700`).
    In this case, the inputs of the `cat` are NOT compute ops, so the `cat` is NOT delegated.
    """
    num_inputs = 2
    quantized_program = to_quantized_edge_program(
        CatModule(dim), [input_shape] * num_inputs, target="imxrt700"
    ).exported_program()

    # Make sure the `Cat` was NOT delegated.
    assert graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert not any(
        "lowered_module" in node.name for node in quantized_program.graph.nodes
    )


@pytest.mark.parametrize(
    "dim, input_shape",
    [
        pytest.param(0, (1, 8, 8, 8), id="axis = 0"),
        pytest.param(0, (8, 8, 8, 8), id="axis = 0, no `1s` in the shape."),
        pytest.param(-4, (1, 8, 8, 8), id="axis = -4"),
        pytest.param(1, (1, 1, 8, 8), id="axis = 1"),
        pytest.param(-3, (1, 1, 8, 8), id="axis = -3"),
        pytest.param(2, (1, 1, 1, 8), id="axis = 2"),
        pytest.param(-2, (1, 1, 1, 8), id="axis = -2"),
    ],
)
def test_cat__context_dependent__imxrt700(dim, input_shape):
    """This test is conjoined with the one above (`test_cat__unsupported__imxrt700`).
    In this case, the inputs of the `cat` are compute ops, so the `cat` is delegated.
    """
    num_inputs = 2
    ep = to_quantized_edge_program(
        AddCatModule(dim), [input_shape] * num_inputs, target="imxrt700"
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(ep.graph, [exir_ops.edge.aten.cat.default])
    assert any("lowered_module" in node.name for node in ep.graph.nodes)


@pytest.mark.parametrize(
    "rank, num_inputs, dim",
    [
        pytest.param(2, 2, 1, id="2D, 2 inputs, dim=1"),
        pytest.param(2, 2, -1, id="2D, 2 inputs, dim=-1"),
        pytest.param(2, 3, 1, id="2D, 3 inputs, dim=1"),
        pytest.param(2, 3, -1, id="2D, 3 inputs, dim=-1"),
        pytest.param(2, 4, -1, id="2D, 4 inputs, dim=-1"),
        pytest.param(3, 2, 1, id="3D, 2 inputs, dim=1"),
        pytest.param(3, 2, -1, id="3D, 2 inputs, dim=-1"),
        pytest.param(3, 5, -1, id="3D, 5 inputs, dim=-2"),
        pytest.param(4, 2, -1, id="4D, 2 inputs, dim=-1"),
        pytest.param(4, 3, 2, id="4D, 3 inputs, dim=2"),
        pytest.param(4, 5, -3, id="4D, 5 inputs, dim=-3"),
    ],
)
def test_cat__different_shapes(dim, num_inputs, rank, mocker):
    input_shape = tuple([2, 8, 8, 8, 8][-rank:])

    # The shape of every input will be different along the concatenated dimension.
    input_shapes = []
    for i in range(num_inputs):
        tmp_shape = list(input_shape)
        tmp_shape[dim] = 8 * (i + 1)  # RT700 requires multiples of 8 for the channels.
        input_shapes.append(tuple(tmp_shape))

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(
        CatModule(dim), input_shapes
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    input_data = {
        i: (np.random.random(shape) * 50).astype(np.int8)
        for i, shape in enumerate(input_shapes)
    }
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        input_data=input_data,
        atol=1,
    )


@pytest.mark.parametrize("dim", [1, -1, -2], ids=lambda dim: f"dim = {dim}")
@pytest.mark.parametrize(
    "num_inputs", [2, 5], ids=lambda num_inputs: f"num_inputs = {num_inputs}"
)
def test_cat__channels_first__different_shapes(dim, num_inputs, mocker):
    input_shape = (2, 8, 6, 8)

    # The shape of every input will be different along the concatenated dimension.
    input_shapes = []
    for i in range(num_inputs):
        tmp_shape = list(input_shape)
        tmp_shape[dim] = 8 * (
            i + 1
        )  # Neutron only supports channels that are multiples of 8 (on RT700).
        input_shapes.append(tuple(tmp_shape))

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    channels = (
        sum(shape[1] for shape in input_shapes) if dim in [1, -3] else input_shape[1]
    )
    quantized_program = to_quantized_edge_program(
        CatConvModule(dim, channels), input_shapes
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    input_data = {
        i: (np.random.random(shape) * 50).astype(np.int8)
        for i, shape in enumerate(input_shapes)
    }
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        input_data=input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        atol=1,
    )


def test_cat__different_shapes__unsupported_channels__imxrt700():
    input_shape = (2, 4, 6, 7)  # (channels % 8) != 0

    num_inputs = 2
    dim = -1

    # The shape of every input will be different along the concatenated dimension.
    input_shapes = []
    for i in range(num_inputs):
        tmp_shape = list(input_shape)
        tmp_shape[dim] = i + 2
        input_shapes.append(tuple(tmp_shape))

    quantized_program = to_quantized_edge_program(
        CatModule(dim), input_shapes, target="imxrt700"
    ).exported_program()

    # Make sure the `Cat` was NOT delegated.
    assert graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert not any(
        "lowered_module" in node.name for node in quantized_program.graph.nodes
    )


def test_cat__force_delegate():
    target = "imxrt700"

    # The Partitioner doesn't know if the `8` or the `1` will become the channels in the IR. Therefore, it would
    #  normally not delegate the `cat`. But we know that the `8` will be the channels, so we can force the delegation.
    input_shape = (8, 1, 8)

    quantized_program = to_quantized_edge_program(
        CatModule(1),
        [input_shape, input_shape],
        target=target,
        custom_delegation_options=CustomDelegationOptions(force_delegate_cat=True),
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)


def test_cat__same_shapes_converter_padding_last_dimension():
    target = "imxrt700"

    # The Converter is capable of padding the last dimension of `cat` with the same input shapes.
    input_shape = (3, 1, 3)

    quantized_program = to_quantized_edge_program(
        CatModule(2),
        [input_shape, input_shape],
        target=target,
        neutron_converter_flavor="SDK_25_09",
        custom_delegation_options=CustomDelegationOptions(),
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)


def test_cat__same_shapes__channels_first__padding_channels():
    target = "imxrt700"

    # The Converter is capable of padding the last dimension of `cat` with the same input shapes.
    input_shape = (1, 2, 3, 4)

    quantized_program = to_quantized_edge_program(
        CatConvModule(1),
        [input_shape, input_shape],
        target=target,
        neutron_converter_flavor="SDK_25_09",
        custom_delegation_options=CustomDelegationOptions(),
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)


def test_cat__same_shapes_converter_padding_middle_dimension():
    target = "imxrt700"

    # The Converter is not capable of padding the middle dimensions of `cat` with the same input shapes.
    input_shape = (3, 1, 3)

    quantized_program = to_quantized_edge_program(
        CatModule(1),
        [input_shape, input_shape],
        target=target,
        custom_delegation_options=CustomDelegationOptions(),
    ).exported_program()

    # Make sure the `Cat` was NOT delegated.
    assert graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert not any(
        "lowered_module" in node.name for node in quantized_program.graph.nodes
    )


def test_cat__format_specific_support__formatless(mocker):
    # The last dim will end up being the channels, as the format is `formatless`.
    # Only the last dim satisfies the Neutron requirements for the channels.
    input_shape = (3, 3, 3, 8)
    num_inputs = 2
    dim = 2

    input_shapes = [input_shape] * num_inputs

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(
        CatModule(dim), input_shapes
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    input_data = {
        i: (np.random.random(shape) * 50).astype(np.int8)
        for i, shape in enumerate(input_shapes)
    }
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        input_data=input_data,
        atol=1,
    )


def test_cat__format_specific_support__channels_first(mocker):
    # The second dim will end up being the channels, as the format is `formatless`.
    # Only the second dim satisfies the Neutron requirements for the channels.
    input_shape = (3, 8, 3, 3)
    num_inputs = 2
    dim = 2

    input_shapes = [input_shape] * num_inputs

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    channels = (
        sum(shape[1] for shape in input_shapes) if dim in [1, -3] else input_shape[1]
    )
    quantized_program = to_quantized_edge_program(
        CatConvModule(dim, channels), input_shapes
    ).exported_program()

    # Make sure the `Cat` was delegated.
    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.cat.default]
    )
    assert any("lowered_module" in node.name for node in quantized_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    input_data = {
        i: (np.random.random(shape) * 50).astype(np.int8)
        for i, shape in enumerate(input_shapes)
    }
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        input_data=input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        atol=1,
    )
