# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Tuple

import torch

from executorch.backends.cadence.aot.passes import (
    RemoveZeroSizedCatArgsPass,
    ReplacePT2DequantWithCadenceDequantPass,
    ReplacePT2QuantWithCadenceQuantPass,
    ReplaceScalarTensorWithFullPass,
    ReplaceSqueezeAndUnsqueezeWithViewPass,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge

from torch.export import export
from torch.export.exported_program import ExportedProgram


# Export the model and lower it to an ExportedProgram (in aten IR)
def export_program(
    model: torch.nn.Module,
    inputs: Tuple[Any, ...],
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


# Export the model and lower it to an EdgeProgramManager (in edge IR).
def export_to_edge(
    model: torch.nn.Module,
    inputs: Tuple[Any, ...],
    dump_graphs: bool = False,
) -> EdgeProgramManager:
    assert isinstance(model, torch.nn.Module), "model should be an nn.Module"

    # Export the model into an ExportedProgram.
    expo_program = export_program(model, inputs)

    if dump_graphs:
        logging.info("Exported graph:")
        expo_program.graph_module.graph.print_tabular()

    # Call to_edge to convert the graph to edge IR.
    # Note: dim_order is skipped (https://github.com/pytorch/executorch/issues/3704)
    edge_prog_manager = to_edge(
        expo_program,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )

    if dump_graphs:
        logging.info("Edge graph:")
        edge_prog_manager.exported_program().graph_module.graph.print_tabular()

    return edge_prog_manager


# Export the model and lower it to an EdgeProgramManager (in edge IR), and
# apply passes specific to Cadence DSP execution.
def export_to_cadence(
    model: torch.nn.Module,
    inputs: Tuple[Any, ...],
    dump_graphs: bool = False,
) -> EdgeProgramManager:
    edge_program_manager = export_to_edge(model, inputs)

    # Run a couple required passes for quant/dequant ops
    cadence_program_manager = edge_program_manager.transform(
        [
            RemoveZeroSizedCatArgsPass(),
            ReplaceScalarTensorWithFullPass(),
            ReplaceSqueezeAndUnsqueezeWithViewPass(),
            ReplacePT2QuantWithCadenceQuantPass(),
            ReplacePT2DequantWithCadenceDequantPass(),
        ]
    )

    return cadence_program_manager
