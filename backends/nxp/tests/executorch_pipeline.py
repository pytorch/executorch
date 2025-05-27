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
    input_shape: tuple,
    operators_not_to_delegate: list[str] = None,
    target="imxrt700",
    neutron_converter_flavor="SDK_25_03",
) -> EdgeProgramManager:
    calibration_inputs = [(torch.randn(input_shape),), (torch.randn(input_shape),)]
    example_input = (torch.ones(*input_shape),)

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
    model: torch.nn.Module, input_shape: tuple
) -> ExecutorchProgramManager:
    edge_program_manager = to_quantized_edge_program(model, input_shape)

    return edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )


def to_edge_program(model: nn.Module, input_shape) -> EdgeProgramManager:
    example_input = (torch.ones(input_shape),)
    exir_program = torch.export.export(model, example_input)
    return exir.to_edge(exir_program)
