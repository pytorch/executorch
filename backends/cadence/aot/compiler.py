# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Tuple

import torch

from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge

from torch.export import export
from torch.export.exported_program import ExportedProgram


def export_program(
    model: torch.nn.Module,
    inputs: Any,
) -> ExportedProgram:
    assert isinstance(model, torch.nn.Module), "model should be an nn.Module"

    # If the model is already a GraphModule (most likely from quantization), call the
    # suggested torch.ao.quantization API instead, which only does dropout and batchnorm.
    if isinstance(model, torch.fx.GraphModule):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        # We don't support training mode. Make it eval
        if hasattr(model, "eval"):
            model.eval()

    # Prevent mkldnn decompositions
    torch._C._set_mkldnn_enabled(False)

    # else: capture the model and return it.
    return export(model, inputs)


# Export the model and lower it it edge IR.
def export_to_edge(
    model: torch.nn.Module,
    inputs: Any,
    dump_graphs: bool = False,
) -> Tuple[EdgeProgramManager, ExportedProgram]:
    # Export the model into an ExportedProgram.
    expo_program = export_program(model, inputs)

    if dump_graphs:
        logging.info(f"Exported graph:\n{expo_program.graph_module.graph}")

    # Call to_edge to convert the graph to edge IR.
    edge_prog_manager = to_edge(
        expo_program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )

    if dump_graphs:
        logging.info(
            f"Edge graph:\n{edge_prog_manager.exported_program().graph_module.graph}"
        )

    return edge_prog_manager, expo_program
