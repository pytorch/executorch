# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable

import eiq_neutron_sdk
import numpy as np
import torch

from executorch import exir
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
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
from executorch.backends.nxp.nxp_backend import (
    core_aten_ops_exception_list,
    generate_neutron_compile_spec,
)
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from torch import memory_format, nn
from torch.export import export
from torchao.quantization.pt2e.quantizer import Quantizer

neutron_target_spec = NeutronTargetSpec(target="imxrt700")


@dataclass
class ModelInputSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    dim_order: memory_format = torch.contiguous_format


def handle_kernel_selection(model_name: str = ""):
    # Collect all kernel selection files in the current working directory
    kernel_selection_files = [
        fname
        for fname in os.listdir(".")
        if re.search("_kernel_selection.*\\.c$", fname)
    ]

    if not kernel_selection_files:
        raise RuntimeError(
            "No kernel_selection files found in the current directory."
        )  # Should never happen.

    output_mask = model_name + "_kernel_selection.c"
    eiq_neutron_sdk.merge_kernel_selection_files(kernel_selection_files, output_mask)

    if not logging.root.isEnabledFor(logging.DEBUG):
        for fname in kernel_selection_files:
            os.remove(fname)
    else:
        logging.debug(
            f"Debug mode enabled, keeping intermediate kernel_selection.c files: {kernel_selection_files}"
        )


def get_random_calibration_inputs(
    input_spec: Iterable[ModelInputSpec], num_samples: int = 4
) -> list[tuple[torch.Tensor, ...]]:
    return [
        tuple([torch.randn(spec.shape, dtype=spec.dtype) for spec in input_spec])
        for _ in range(num_samples)
    ]


def _get_default_quantizer(target_spec: NeutronTargetSpec, use_qat: bool) -> Quantizer:
    return NeutronQuantizer(target_spec, is_qat=use_qat)


def to_model_input_spec(
    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]]
) -> tuple[ModelInputSpec, ...]:
    match input_spec:
        case _ if isinstance(input_spec, Iterable) and all(
            isinstance(spec, ModelInputSpec) for spec in input_spec
        ):
            return tuple(input_spec)
        case tuple() if all(isinstance(spec, int) for spec in input_spec):
            return (ModelInputSpec(input_spec),)
        case list() if all(
            isinstance(input_shape, tuple) for input_shape in input_spec
        ):
            return tuple(ModelInputSpec(spec) for spec in input_spec)
        case _:
            raise TypeError(f"Unsupported type {type(input_spec)}")


GetCalibrationInputsFn = Callable[
    [tuple[ModelInputSpec, ...]], Iterable[tuple[torch.Tensor, ...]]
]


def get_calibration_inputs_fn_from_dataset_dir(dataset_dir) -> GetCalibrationInputsFn:
    def _nested(
        input_spec: tuple[ModelInputSpec, ...]
    ) -> Iterable[tuple[torch.Tensor, ...]]:
        data = sorted(os.listdir(dataset_dir))
        inputs_needed = len(input_spec)

        for path in data:
            path = os.path.join(dataset_dir, path)
            files = []

            if os.path.isdir(path):
                files = [os.path.join(path, x) for x in sorted(os.listdir(path))]
            else:
                files.append(path)

            input_data = []
            for idx, file in enumerate(files):
                if len(input_data) == inputs_needed:
                    break

                tensor = np.fromfile(
                    file, dtype=torch_type_to_numpy_type(input_spec[idx].dtype)
                ).reshape(input_spec[idx].shape)
                input_data += (torch.from_numpy(tensor),)
                continue

            if len(input_data) < inputs_needed:
                continue

            yield tuple(input_data)

    return _nested


def _get_example_input(
    input_spec: tuple[ModelInputSpec, ...]
) -> tuple[torch.Tensor, ...]:
    example_input = []
    for spec in input_spec:
        match spec.dim_order:
            case torch.contiguous_format:
                sample = torch.ones(spec.shape, dtype=spec.dtype)
            case torch.channels_last:
                sample = torch.ones(spec.shape, dtype=spec.dtype).to(
                    memory_format=torch.channels_last
                )
            case _:
                raise ValueError(f"Unsupported dim_order: {spec.dim_order}")
        # noinspection PyUnboundLocalVariable
        example_input.append(sample)

    return tuple(example_input)


def to_quantized_edge_program(
    model: torch.nn.Module,
    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]],
    operators_not_to_delegate: list[str] = None,
    get_calibration_inputs_fn: GetCalibrationInputsFn = get_random_calibration_inputs,
    target: str = "imxrt700",
    use_qat: bool = False,
    train_fn: Callable[[torch.fx.GraphModule], None] | None = None,
    remove_quant_io_ops: bool = False,
    custom_delegation_options: CustomDelegationOptions = CustomDelegationOptions(),  # noqa B008
    get_quantizer_fn: Callable[[], Quantizer] | None = None,
    use_neutron_for_format_conversion: bool = True,
    use_quant_state_dict: bool = True,
    fetch_constants_to_sram: bool = False,
    dump_kernel_selection_code: bool = False,
    use_new_flow_neutron_c: bool = False,
    delegate_to_npu=True,
) -> EdgeProgramManager:
    _neutron_target_spec = NeutronTargetSpec(target)
    custom_delegation_options.use_new_flow_neutron_c = use_new_flow_neutron_c
    if get_quantizer_fn is None:
        get_quantizer_fn = partial(
            _get_default_quantizer, _neutron_target_spec, use_qat
        )
    input_spec = to_model_input_spec(input_spec)
    calibration_inputs = get_calibration_inputs_fn(input_spec)
    example_input = _get_example_input(input_spec)

    # Make sure the model is in the evaluation mode.
    model.eval()

    exir_program_aten = torch.export.export(model, example_input, strict=True)

    exir_program_aten__module_quant = calibrate_and_quantize(
        model=exir_program_aten,
        calibration_inputs=calibration_inputs,
        quantizer=get_quantizer_fn(),
        is_qat=use_qat,
        train_fn=train_fn,
    )

    # List of operators to not decompose during the lowering.
    preserve_ops = [torch.ops.aten.prelu.default]
    compile_spec = generate_neutron_compile_spec(
        target,
        operators_not_to_delegate=operators_not_to_delegate,
        use_neutron_for_format_conversion=use_neutron_for_format_conversion,
        fetch_constants_to_sram=fetch_constants_to_sram,
        dump_kernel_selection_code=dump_kernel_selection_code,
        use_new_flow_neutron_c=use_new_flow_neutron_c,
    )
    post_quant_state_dict = (
        exir_program_aten__module_quant.state_dict() if use_quant_state_dict else None
    )
    if delegate_to_npu:
        partitioners = [
            NeutronPartitioner(
                compile_spec,
                _neutron_target_spec,
                custom_delegation_options,
                post_quant_state_dict,
                preserve_ops=preserve_ops,
            )
        ]
    else:
        partitioners = []

    edge_program_manager = to_edge_transform_and_lower(
        export(exir_program_aten__module_quant, example_input, strict=True),
        transform_passes=NeutronEdgePassManager(),
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _core_aten_ops_exception_list=core_aten_ops_exception_list,
        ),
    )

    if remove_quant_io_ops:
        edge_program_manager = edge_program_manager.transform(
            [RemoveIOQuantOpsPass(edge_program_manager=edge_program_manager)]
        )

    edge_program_manager = edge_program_manager.transform(
        NeutronEdgePassManager([RemoveAdditionalQDQClustersPass()])
    )

    if dump_kernel_selection_code:
        handle_kernel_selection()

    return edge_program_manager


def to_quantized_executorch_program(
    model: torch.nn.Module,
    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]],
    use_qat: bool = False,
    train_fn: Callable[[torch.fx.GraphModule], None] | None = None,
    use_neutron_for_format_conversion: bool = True,
    dataset_dir: str | None = None,
    delegate_to_npu=True,
    use_new_flow_neutron_c: bool = False,
) -> ExecutorchProgramManager:
    if dataset_dir:
        # Extract calibration data from a directory.
        get_calibration_inputs_fn = {
            "get_calibration_inputs_fn": get_calibration_inputs_fn_from_dataset_dir(
                dataset_dir
            )
        }
    else:
        get_calibration_inputs_fn = {}  # Use default parameter value.

    edge_program_manager = to_quantized_edge_program(
        model,
        input_spec,
        use_qat=use_qat,
        train_fn=train_fn,
        use_neutron_for_format_conversion=use_neutron_for_format_conversion,
        delegate_to_npu=delegate_to_npu,
        use_new_flow_neutron_c=use_new_flow_neutron_c,
        **get_calibration_inputs_fn,
    )

    return edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )


def to_edge_program(
    model: nn.Module,
    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]],
) -> EdgeProgramManager:
    example_input = _get_example_input(to_model_input_spec(input_spec))

    # Make sure the model is in the evaluation mode.
    model.eval()

    exir_program = torch.export.export(model, example_input)
    return exir.to_edge(exir_program)
