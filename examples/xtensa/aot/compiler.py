# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable

import torch

from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge

from torch.export import export
from torch.export.exported_program import ExportedProgram


def export_program(
    model: Callable,
    inputs: Any,
    pt2_quant: bool = False,
) -> ExportedProgram:
    # we don't support training mode. Make it eval
    if hasattr(model, "eval"):
        if pt2_quant:
            # pyre-fixme[6]: Incompatible parameter type.
            torch.ao.quantization.move_exported_model_to_eval(model)
        else:
            # pyre-fixme[16]: Anonymous callable has no attribute `eval`.
            model.eval()

    # if it's already an ExportedProgram, just return it
    if isinstance(model, ExportedProgram):
        return model

    assert isinstance(model, torch.nn.Module), "model should be an nn.Module"

    # Prevent mkldnn decompositions
    torch._C._set_mkldnn_enabled(False)

    # else: capture the model and return it.
    return export(model, inputs)


# Export the model and lower it it edge IR.
def export_to_edge(
    model: Callable,
    inputs: Any,
    pt2_quant: bool = False,
    dump_graphs: bool = False,
) -> EdgeProgramManager:
    # Export the model into an ExportedProgram.
    expo_program = export_program(model, inputs, pt2_quant)

    if dump_graphs:
        logging.info(
            f"Exported graph:\n{expo_program.graph_module.graph}"
        )

    # Call to_edge to convert the graph to edge IR.
    edge_prog_manager = to_edge(
        expo_program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )

    if dump_graphs:
        logging.info(
            f"Edge graph:\n{edge_prog_manager.exported_program().graph_module.graph}"
        )

    return edge_prog_manager
