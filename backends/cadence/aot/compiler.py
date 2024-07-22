# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import torch

from executorch.backends.cadence.aot.passes import (
    RemoveZeroSizedCatArgsPass,
    ReplacePT2DequantWithCadenceDequantPass,
    ReplacePT2QuantWithCadenceQuantPass,
    ReplaceScalarTensorWithFullPass,
    ReplaceSqueezeAndUnsqueezeWithViewPass,
)
from executorch.backends.cadence.aot.quantizer.fusion_pass import QuantFusion
from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceAtenQuantizer,
    CadenceQuantizer,
)
from executorch.backends.cadence.aot.utils import model_is_quantized
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from pyre_extensions import assert_is_instance
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.pt2e.export_utils import model_is_exported
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from torch.export import export
from torch.export.exported_program import ExportedProgram


def quantize_pt2(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
) -> torch.fx.GraphModule:
    """
    Instantiate the CadenceQuantizer (PTQ), prepare, convert and fuse the model.
    Returns a GraphModule with the quantized model.
    """
    # Quantizer
    quantizer = CadenceQuantizer()

    # Export with dynamo
    model_exp = capture_pre_autograd_graph(model, inputs)

    # Decompose SDPA
    DecomposeScaledDotProductAttention(False)(model_exp)

    # Prepare
    prepared_model = prepare_pt2e(model_exp, quantizer)

    # Calibrate
    prepared_model(*inputs)

    # Convert
    converted_model = convert_pt2e(prepared_model)

    # Get patterns and apply fusion of dq -> op -> q to qop
    patterns = [
        assert_is_instance(q, CadenceAtenQuantizer).pattern
        for q in quantizer.quantizers
    ]
    QuantFusion(patterns)(converted_model)

    return converted_model


# Export the model and lower it to an ExportedProgram (in aten IR)
def export_program(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
) -> ExportedProgram:
    assert isinstance(model, torch.nn.Module), "model should be an nn.Module"

    # We don't support training mode. Make the model inference mode by
    # calling model.eval() or an equivalent call for quantized models.
    # GraphModules cannot call eval(), so we skip them.
    if not isinstance(model, torch.fx.GraphModule):
        if hasattr(model, "eval"):
            model.eval()
    else:
        # If the model is quantized, call the suggested torch.ao.quantization API
        # which only does dropout and batchnorm.
        if model_is_quantized(model):
            torch.ao.quantization.move_exported_model_to_eval(model)
        else:
            # If we get a GraphModule which is _not_ quantized, then it should already
            # have been exported.
            assert model_is_exported(model), "model should be from an ExportedProgram"

    # Prevent mkldnn decompositions
    torch._C._set_mkldnn_enabled(False)

    # else: capture the model and return it.
    return export(model, inputs)


# Export the model and lower it to an EdgeProgramManager (in edge IR).
def export_to_edge(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
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
    inputs: tuple[object, ...],
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
