# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable

import torch

from executorch import exir
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.edge_passes.remove_io_quant_ops_pass import (
    RemoveIOQuantOpsPass,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
)
from executorch.extension.export_util.utils import export_to_edge
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


@dataclass
class ModelInputSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32


def _quantize_model(model, calibration_inputs: list[tuple[torch.Tensor, ...]]):
    quantizer = NeutronQuantizer()

    m = prepare_pt2e(model, quantizer)
    for data in calibration_inputs:
        m(*data)
    m = convert_pt2e(m)

    return m


def get_random_calibration_inputs(
    input_spec: tuple[ModelInputSpec, ...]
) -> list[tuple[torch.Tensor, ...]]:
    return [
        tuple([torch.randn(spec.shape, dtype=spec.dtype) for spec in input_spec])
        for _ in range(4)
    ]


def to_model_input_spec(
    input_spec: tuple[ModelInputSpec, ...] | tuple[int, ...] | list[tuple[int, ...]]
) -> tuple[ModelInputSpec, ...]:

    if isinstance(input_spec, tuple) and all(
        isinstance(spec, ModelInputSpec) for spec in input_spec
    ):
        return input_spec

    elif isinstance(input_spec, tuple) and all(
        isinstance(spec, int) for spec in input_spec
    ):
        return (ModelInputSpec(input_spec),)

    elif isinstance(input_spec, list) and all(
        isinstance(input_shape, tuple) for input_shape in input_spec
    ):
        return tuple([ModelInputSpec(spec) for spec in input_spec])
    else:
        raise TypeError(f"Unsupported type {type(input_spec)}")


def to_quantized_edge_program(
    model: torch.nn.Module,
    input_spec: tuple[ModelInputSpec, ...] | tuple[int, ...] | list[tuple[int, ...]],
    operators_not_to_delegate: list[str] = None,
    get_calibration_inputs_fn: Callable[
        [tuple[ModelInputSpec, ...]], list[tuple[torch.Tensor, ...]]
    ] = get_random_calibration_inputs,
    target="imxrt700",
    neutron_converter_flavor="SDK_25_06",
    remove_quant_io_ops=False,
    custom_delegation_options=CustomDelegationOptions(),  # noqa B008
) -> EdgeProgramManager:
    calibration_inputs = get_calibration_inputs_fn(to_model_input_spec(input_spec))

    example_input = calibration_inputs[0]

    # Make sure the model is in the evaluation mode.
    model.eval()

    exir_program_aten = torch.export.export(model, example_input, strict=True)

    exir_program_aten__module_quant = _quantize_model(
        exir_program_aten.module(), calibration_inputs
    )

    edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_program_manager = export_to_edge(
        exir_program_aten__module_quant,
        example_input,
        edge_compile_config=edge_compile_config,
    )

    edge_program_manager = NeutronEdgePassManager()(edge_program_manager)

    compile_spec = generate_neutron_compile_spec(
        target,
        operators_not_to_delegate=operators_not_to_delegate,
        neutron_converter_flavor=neutron_converter_flavor,
    )
    partitioner = NeutronPartitioner(compile_spec, custom_delegation_options)
    edge_program_manager = edge_program_manager.to_backend(partitioner)

    if remove_quant_io_ops:
        edge_program_manager = edge_program_manager.transform(
            [RemoveIOQuantOpsPass(edge_program_manager=edge_program_manager)]
        )

    return edge_program_manager


def to_quantized_executorch_program(
    model: torch.nn.Module,
    input_spec: tuple[ModelInputSpec, ...] | tuple[int, ...] | list[tuple[int, ...]],
) -> ExecutorchProgramManager:
    edge_program_manager = to_quantized_edge_program(model, input_spec)

    return edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )


def to_edge_program(
    model: nn.Module,
    input_spec: tuple[ModelInputSpec, ...] | tuple[int, ...] | list[tuple[int, ...]],
) -> EdgeProgramManager:
    calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_spec))

    example_input = calibration_inputs[0]

    # Make sure the model is in the evaluation mode.
    model.eval()

    exir_program = torch.export.export(model, example_input)
    return exir.to_edge(exir_program)
