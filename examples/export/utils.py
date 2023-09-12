# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import Tuple

import executorch.exir as exir

import torch
import torch._export as export
from executorch.exir.program import ExirExportedProgram
from executorch.exir.tracer import Value


_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True)

# Explicitly force the activation of the IR validator
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=True,
)


def _to_core_aten(
    model: torch.fx.GraphModule,
    example_inputs: Tuple[Value, ...],
    capture_config=_CAPTURE_CONFIG,
) -> ExirExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_exir_ep = exir.capture(model, example_inputs, capture_config)
    logging.info(f"Core ATen graph:\n{core_aten_exir_ep.exported_program.graph}")
    return core_aten_exir_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExirExportedProgram,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
) -> ExirExportedProgram:
    edge = core_aten_exir_ep.to_edge(edge_compile_config)
    logging.info(f"Exported graph:\n{edge.exported_program.graph}")
    return edge


def export_to_edge(
    model: torch.fx.GraphModule,
    example_inputs: Tuple[Value, ...],
    capture_config=_CAPTURE_CONFIG,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
) -> ExirExportedProgram:
    core_aten_exir_ep = _to_core_aten(model, example_inputs, capture_config)
    return _core_aten_to_edge(core_aten_exir_ep, edge_compile_config)


def export_to_exec_prog(
    model,
    example_inputs,
    capture_config=_CAPTURE_CONFIG,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
    backend_config=None,
):
    m = model.eval()
    # pre-autograd export. eventually this will become torch.export
    m = export.capture_pre_autograd_graph(m, example_inputs)

    core_aten_exir_ep = _to_core_aten(m, example_inputs)

    edge_m = _core_aten_to_edge(core_aten_exir_ep, edge_compile_config)

    exec_prog = edge_m.to_executorch(backend_config)
    return exec_prog


def save_pte_program(buffer, model_name):
    filename = f"{model_name}.pte"
    try:
        with open(filename, "wb") as file:
            file.write(buffer)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")
