# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch import exir
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def _quantize_model(model, calibration_inputs: list[tuple[torch.Tensor]]):
    quantizer = NeutronQuantizer()

    m = prepare_pt2e(model, quantizer)
    for data in calibration_inputs:
        m(*data)
    m = convert_pt2e(m)

    return m


def to_quantized_edge_program(
    model: torch.nn.Module,
    input_shapes: tuple[int] | list[tuple[int]],
    operators_not_to_delegate: list[str] = None,
    target="imxrt700",
    neutron_converter_flavor="SDK_25_03",
) -> EdgeProgramManager:
    if isinstance(input_shapes, list):
        assert all(isinstance(input_shape, tuple) for input_shape in input_shapes), (
            "For multiple inputs, provide" " list[tuple[int]]."
        )

    random_tensors = (
        (torch.randn(input_shapes),)
        if type(input_shapes) is tuple
        else tuple(torch.randn(input_shape) for input_shape in input_shapes)
    )
    calibration_inputs = [random_tensors, random_tensors]
    example_input = (
        (torch.ones(input_shapes),)
        if type(input_shapes) is tuple
        else tuple(torch.ones(input_shape) for input_shape in input_shapes)
    )

    exir_program_aten = torch.export.export_for_training(
        model, example_input, strict=True
    )

    exir_program_aten__module_quant = _quantize_model(
        exir_program_aten.module(), calibration_inputs
    )

    compile_spec = generate_neutron_compile_spec(
        target,
        operators_not_to_delegate=operators_not_to_delegate,
        neutron_converter_flavor=neutron_converter_flavor,
    )
    partitioner = NeutronPartitioner(compile_spec)
    edge_program_manager = to_edge_transform_and_lower(
        torch.export.export(
            exir_program_aten__module_quant, example_input, strict=True
        ),
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    return edge_program_manager


def to_quantized_executorch_program(
    model: torch.nn.Module, input_shapes: tuple[int] | list[tuple[int]]
) -> ExecutorchProgramManager:
    edge_program_manager = to_quantized_edge_program(model, input_shapes)

    return edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )


def to_edge_program(
    model: nn.Module, input_shapes: tuple[int] | list[tuple[int]]
) -> EdgeProgramManager:
    if isinstance(input_shapes, list):
        assert all(isinstance(input_shape, tuple) for input_shape in input_shapes), (
            "For multiple inputs, provide" " list[tuple[int]]."
        )

    example_input = (
        (torch.ones(input_shapes),)
        if type(input_shapes) is tuple
        else tuple(torch.ones(input_shape) for input_shape in input_shapes)
    )
    exir_program = torch.export.export(model, example_input)
    return exir.to_edge(exir_program)
