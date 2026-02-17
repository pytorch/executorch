# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 - 2026 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torchao.quantization.pt2e import move_exported_model_to_eval
from torch.export import export, ExportedProgram
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e

import executorch.exir as exir
from executorch.backends.nxp.tests_models.model_input_spec import ModelInputSpec
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import NeutronEdgePassManager
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.devtools.visualization.visualization_utils import visualize_with_clusters
from executorch.exir import (
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
    EdgeCompileConfig,
)
from executorch.exir.capture._config import EdgeCompileConfig
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.tracer import Value

_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=True,
    _skip_dim_order=True,  # TODO(T189114319): Reuse dim order op after solving the ios oss issue
)


def to_quantized_edge_program(model: torch.nn.Module, input_spec: list[ModelInputSpec], dataset_dir,
                              delegate_to_npu=True, use_qat: bool = False) -> EdgeProgramManager:
    assert isinstance(input_spec, list) and all([isinstance(spec, ModelInputSpec) for spec in input_spec]), \
        "Input_spec must be a list of ModelInputSpec."

    example_input = []
    for spec in input_spec:
        match spec.dim_order:
            case torch.contiguous_format:
                sample = torch.ones(spec.shape, dtype=spec.dtype)
            case torch.channels_last:
                sample = torch.ones(spec.shape, dtype=spec.dtype).to(memory_format=torch.channels_last)
            case _:
                raise ValueError(f"Unsupported dim_order: {spec.dim_order}")
        # noinspection PyUnboundLocalVariable
        example_input.append(sample)

    example_input = tuple(example_input)

    exir_program_aten = torch.export.export(model, example_input, strict=True)
    module = exir_program_aten.module()

    neutron_target_spec = NeutronTargetSpec(
        target="imxrt700", neutron_converter_flavor="SDK_25_12"
    )

    # Quantize model
    quantizer = NeutronQuantizer(neutron_target_spec=neutron_target_spec, is_qat=use_qat)
    if use_qat:
        m = prepare_qat_pt2e(module, quantizer)
        m = move_exported_model_to_eval(m)
    else:
        m = prepare_pt2e(module, quantizer)

    data = sorted(os.listdir(dataset_dir))
    inputs_needed = len(input_spec)

    # If the model is single-input, the following directory structure is used:
    #   dataset_dir/data.bin (data.bin is *the* input)
    # Else, if multi-input, the following directory structure is used:
    #   dataset_dir/data/{.+}.bin (each .bin file is an input)

    input_data = []
    for path in data:
        path = os.path.join(dataset_dir, path)
        files = []

        if os.path.isdir(path):
            files = [os.path.join(path, x) for x in sorted(os.listdir(path))]
        else:
            files.append(path)

        for idx, file in enumerate(files):
            if len(input_data) == inputs_needed:
                break

            tensor = np.fromfile(file, dtype=input_spec[idx].type).reshape(input_spec[idx].shape)
            input_data += (torch.from_numpy(tensor),)
            continue

        if len(input_data) < inputs_needed:
            continue

        m(*input_data)
        input_data.clear()

    exir_program_aten_quant = convert_pt2e(m)

    # To ATen
    core_aten_ep = _to_core_aten(exir_program_aten_quant, example_input, None, verbose=True)

    partitioners = (
        [
            NeutronPartitioner(
                generate_neutron_compile_spec("imxrt700", "SDK_25_12"),
                neutron_target_spec=neutron_target_spec,
                post_quantization_state_dict=exir_program_aten_quant.state_dict(),
            )
        ]
    ) if delegate_to_npu else []

    edge_program_manager = to_edge_transform_and_lower(
        core_aten_ep,
        transform_passes=NeutronEdgePassManager(),
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(),
    )

    return edge_program_manager


def to_quantized_executorch_program(model: torch.nn.Module, input_spec, dataset_dir: str,
                                    delegate_to_npu=True, use_qat: bool = False) -> ExecutorchProgramManager:
    edge_program_manager = to_quantized_edge_program(
        model, input_spec, dataset_dir, delegate_to_npu, use_qat=use_qat
    )

    return edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=False
        )
    )


def _to_core_aten(
        model: Union[torch.fx.GraphModule, torch.nn.Module],
        example_inputs: Tuple[Value, ...],
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        verbose=True,
) -> ExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule) and not isinstance(
            model, torch.nn.Module
    ):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_ep = export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    if verbose:
        logging.info(f"Core ATen graph:\n{core_aten_ep.graph}")
    return core_aten_ep


def save_pte_program(
        prog: ExecutorchProgramManager, model_name: str, output_dir: str = ""
) -> str:
    if model_name.endswith(".pte"):
        filename = model_name
        visualize_file_name = f"{model_name}.json"
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")
        visualize_file_name = os.path.join(output_dir, f"{model_name}.json")
    try:
        with open(filename, "wb") as file:
            prog.write_to_file(file)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")

    visualize_with_clusters(prog.exported_program(), visualize_file_name, False)
    return filename
