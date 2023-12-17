# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from typing import Tuple, Union

import executorch.exir as exir

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.program._program import ExecutorchProgram, ExirExportedProgram
from executorch.exir.tracer import Value
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export import export, ExportedProgram


_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=True,
)


def _to_core_aten(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
) -> ExirExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    # core_aten_ep = export(model, example_inputs)
    core_aten_ep = exir.capture(
        model, example_inputs, exir.CaptureConfig(enable_aot=True)
    )
    logging.info(f"Core ATen graph:\n{core_aten_ep.exported_program.graph}")
    return core_aten_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExirExportedProgram,
    edge_compile_config=None,
) -> ExirExportedProgram:
    if not edge_compile_config:
        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,  # quant ops currently break ir verification
        )
    edge = core_aten_exir_ep.to_edge(
        exir.EdgeCompileConfig(
            _check_ir_validity=False,
        )
    )
    edge.exported_program = to_backend(edge.exported_program, XnnpackPartitioner())
    logging.info(f"Exported graph:\n{edge.exported_program.graph}")
    return edge


def export_to_edge(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    edge_compile_config=_EDGE_COMPILE_CONFIG,
) -> EdgeProgramManager:
    core_aten_ep = _to_core_aten(model, example_inputs)
    return _core_aten_to_edge(core_aten_ep, edge_compile_config)


def export_to_exec_prog(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    edge_compile_config=_EDGE_COMPILE_CONFIG,
    backend_config=None,
) -> ExecutorchProgram:
    m = model.eval()
    # pre-autograd export. eventually this will become torch.export
    m = capture_pre_autograd_graph(m, example_inputs)

    quantize = False
    if quantize:
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        prepared_gm = prepare_pt2e(m, quantizer)
        logging.info("get_egoru_backbone_int8_pt2 before calibration")
        with torch.no_grad():
            prepared_gm(*example_inputs)

        logging.info("before convert")
        m = convert_pt2e(prepared_gm)

    core_aten_ep = _to_core_aten(m, example_inputs)

    edge_m = _core_aten_to_edge(core_aten_ep, edge_compile_config)

    exec_prog = edge_m.to_executorch(backend_config)
    return exec_prog


def save_pte_program(buffer: bytes, model_name: str, output_dir: str = "") -> None:
    filename = os.path.join(output_dir, f"{model_name}.pte")
    try:
        with open(filename, "wb") as file:
            file.write(buffer)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")
