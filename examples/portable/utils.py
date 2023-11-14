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
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.tracer import Value
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram


_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)


def _to_core_aten(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
) -> ExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_ep = export(model, example_inputs)
    logging.info(f"Core ATen dialect graph:\n{core_aten_ep.graph}")
    return core_aten_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExportedProgram,
    edge_compile_config=None,
) -> EdgeProgramManager:
    if not edge_compile_config:
        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,  # quant ops currently break ir verification
        )
    edge_manager: EdgeProgramManager = to_edge(
        core_aten_exir_ep, compile_config=edge_compile_config
    )
    logging.info(f"Edge dialect graph:\n{edge_manager.exported_program().graph}")
    return edge_manager


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
) -> ExecutorchProgramManager:
    m = model.eval()
    # pre-autograd export. eventually this will become torch.export
    m = capture_pre_autograd_graph(m, example_inputs)

    core_aten_ep = _to_core_aten(m, example_inputs)

    edge_m = _core_aten_to_edge(core_aten_ep, edge_compile_config)

    exec_prog = edge_m.to_executorch(backend_config)
    logging.info(f"ExecuTorch program graph:\n{exec_prog.exported_program().graph}")
    return exec_prog


def save_pte_program(buffer: bytes, model_name: str, output_dir: str = "") -> None:
    filename = os.path.join(output_dir, f"{model_name}.pte")
    try:
        with open(filename, "wb") as file:
            file.write(buffer)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")
