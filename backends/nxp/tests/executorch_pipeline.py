# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

from executorch import exir
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.edge_passes.remove_additional_quantize_dequantize_nodes_pass import (
    RemoveAdditionalQDQClustersPass,
)
from executorch.backends.nxp.edge_passes.remove_io_quant_ops_pass import (
    RemoveIOQuantOpsPass,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from torch import nn
from torch.export import export
from torchao.quantization.pt2e.quantizer import Quantizer

neutron_converter_flavor = "SDK_25_12"
neutron_target_spec = NeutronTargetSpec(
    target="imxrt700", neutron_converter_flavor=neutron_converter_flavor
)


@dataclass
class ModelInputSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32


def get_random_calibration_inputs(
    input_spec: tuple[ModelInputSpec, ...]
) -> list[tuple[torch.Tensor, ...]]:
    return [
        tuple([torch.randn(spec.shape, dtype=spec.dtype) for spec in input_spec])
        for _ in range(4)
    ]


def _get_default_quantizer(target_spec: NeutronTargetSpec, use_qat: bool) -> Quantizer:
    return NeutronQuantizer(target_spec, is_qat=use_qat)


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
    neutron_converter_flavor=neutron_converter_flavor,
    use_qat=False,
    remove_quant_io_ops=False,
    custom_delegation_options=CustomDelegationOptions(),  # noqa B008
    get_quantizer_fn=None,
    use_neutron_for_format_conversion=True,
) -> EdgeProgramManager:
    _neutron_target_spec = NeutronTargetSpec(target, neutron_converter_flavor)
    if get_quantizer_fn is None:
        get_quantizer_fn = partial(
            _get_default_quantizer, _neutron_target_spec, use_qat
        )

    calibration_inputs = get_calibration_inputs_fn(to_model_input_spec(input_spec))
    example_input = calibration_inputs[0]

    # Make sure the model is in the evaluation mode.
    model.eval()

    exir_program_aten = torch.export.export(model, example_input, strict=True)

    exir_program_aten__module_quant = calibrate_and_quantize(
        model=exir_program_aten,
        calibration_inputs=calibration_inputs,
        quantizer=get_quantizer_fn(),
        is_qat=use_qat,
    )

    compile_spec = generate_neutron_compile_spec(
        target,
        operators_not_to_delegate=operators_not_to_delegate,
        neutron_converter_flavor=neutron_converter_flavor,
        use_neutron_for_format_conversion=use_neutron_for_format_conversion,
    )
    partitioners = [
        NeutronPartitioner(
            compile_spec, _neutron_target_spec, custom_delegation_options
        )
    ]

    edge_program_manager = to_edge_transform_and_lower(
        export(exir_program_aten__module_quant, example_input, strict=True),
        transform_passes=NeutronEdgePassManager(),
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    if remove_quant_io_ops:
        edge_program_manager = edge_program_manager.transform(
            [RemoveIOQuantOpsPass(edge_program_manager=edge_program_manager)]
        )

    edge_program_manager = edge_program_manager.transform(
        NeutronEdgePassManager([RemoveAdditionalQDQClustersPass()])
    )

    return edge_program_manager


def to_quantized_executorch_program(
    model: torch.nn.Module,
    input_spec: tuple[ModelInputSpec, ...] | tuple[int, ...] | list[tuple[int, ...]],
    use_qat: bool = False,
    use_neutron_for_format_conversion: bool = True,
) -> ExecutorchProgramManager:
    edge_program_manager = to_quantized_edge_program(
        model,
        input_spec,
        use_qat=use_qat,
        use_neutron_for_format_conversion=use_neutron_for_format_conversion,
    )

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
