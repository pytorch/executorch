# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from pathlib import Path
from typing import Callable, cast, Optional

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.quantizer.fusion_pass import QuantFusion
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceQuantizer

from executorch.backends.cadence.aot.replace_ops import ReplaceSafeSoftmaxWithSoftmax
from executorch.backends.cadence.aot.utils import model_gm_has_SDPA, model_is_quantized
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from executorch.devtools import generate_etrecord
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
)
from executorch.exir.pass_base import PassResult
from torch._inductor.decomposition import remove_decompositions
from torch.ao.quantization.pt2e.export_utils import model_is_exported
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from torch.export import export
from torch.export.exported_program import ExportedProgram

from .passes import get_cadence_passes

from .utils import print_ops_info


# Note: this is not meant as a primary API since it can create inconsistencies
# if the quantizer here is different from the quantizer used to convert. It is
# however useful for unit tests to separate the converted model from the fused
# model, to be able to get reference numerics.
# If this does not apply, please use quantize_and_fuse_pt2 instead.
def convert_pt2(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    quantizer: CadenceQuantizer,
) -> torch.fx.GraphModule:
    """
    Prepare and convert a model using the given quantizer.
    The quantizer must be supplied and be the same as the one used to
    fuse the model later, if applicable. If you do not expect that behavior,
    please use quantize_and_fuse_pt2 instead, which will instantiate a
    default quantizer for you if needed.
    Returns a GraphModule with the converted model.
    """

    # Get default decompositions
    decomp_table = torch.export.default_decompositions()
    # Select ops to keep
    ops_to_keep = [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
    ]
    # Remove decompositions for the ops we want to keep
    # pyre-fixme[6]: For 1st argument expected `Dict[typing.Callable[..., typing.Any
    remove_decompositions(decomp_table, ops_to_keep)
    # Export with dynamo
    model_gm = (
        torch.export.export_for_training(model, inputs)
        .run_decompositions(decomp_table)
        .module()
    )

    if model_gm_has_SDPA(model_gm):
        # Decompose SDPA
        DecomposeScaledDotProductAttention(False)(model_gm)

        # Swap _safe_softmax with _softmax (see https://github.com/pytorch/pytorch/pull/133882
        # for details).
        result = ReplaceSafeSoftmaxWithSoftmax()(model_gm)
        assert result is not None
        model_gm = result.graph_module

    # Prepare
    prepared_model = prepare_pt2e(model_gm, quantizer)

    # Calibrate
    prepared_model(*inputs)

    # Convert
    converted_model = convert_pt2e(prepared_model)

    return converted_model


# Note: this is not meant as a primary API since it can create inconsistencies
# if the quantizer here is different from the quantizer used to convert. It is
# however useful for unit tests to separate the converted model from the fused
# model, to be able to get reference numerics.
# If this does not apply, please use quantize_and_fuse_pt2 instead.
def fuse_pt2(
    converted_graph_module: torch.fx.GraphModule,
    quantizer: CadenceQuantizer,
) -> torch.fx.GraphModule:
    """
    Fuse a converted graph module using the given quantizer.
    The quantizer must be the same as the one used to convert the model.
    If you do not expect that behavior, please use quantize_and_fuse_pt2 instead,
    which will instantiate a default quantizer for you if needed.
    Returns a GraphModule with the fused model.
    """
    # Get patterns and apply fusion of dq -> op -> q to qop
    # pyre-ignore[16]: no attribute
    patterns = [q.pattern for q in quantizer.quantizers]
    QuantFusion(patterns)(converted_graph_module)

    return converted_graph_module


# Note: this is the one-liner API to quantize and fuse a model.
def quantize_pt2(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    quantizer: Optional[CadenceQuantizer] = None,
) -> torch.fx.GraphModule:
    """
    Prepare, convert and fuse the model using the given quantizer.
    Returns a GraphModule with the quantized model.
    """
    # Quantizer
    if not quantizer:
        quantizer = CadenceQuantizer()

    # Get converted graph module
    converted_gm = convert_pt2(model, inputs, quantizer)

    # Get fused model
    fused_gm = fuse_pt2(converted_gm, quantizer)

    return fused_gm


# Export the model and lower it to an ExportedProgram (in aten IR)
def export_program(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    dump_graphs: bool = False,
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
    expo_program = export(model, inputs)

    if dump_graphs:
        logging.info("Exported graph:")
        expo_program.graph_module.graph.print_tabular()

    return expo_program


# Export the model and lower it to an EdgeProgramManager (in edge IR).
def export_to_edge(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    dump_graphs: bool = False,
) -> EdgeProgramManager:
    assert isinstance(model, torch.nn.Module), "model should be an nn.Module"

    # Export the model into an ExportedProgram.
    expo_program = export_program(model, inputs, dump_graphs=dump_graphs)

    # Call to_edge to convert the graph to edge IR.
    # Note: dim_order is skipped (https://github.com/pytorch/executorch/issues/3704)
    edge_prog_manager = to_edge(
        expo_program,
        compile_config=EdgeCompileConfig(
            _skip_dim_order=True,
            # Allow specific non-core aten ops in the IR.
            _core_aten_ops_exception_list=[
                torch.ops.aten.linear.default,
                torch.ops.aten.native_batch_norm.default,
                torch.ops.aten.linalg_vector_norm.default,
                torch.ops.aten.unfold.default,
                torch.ops.aten.angle.default,
            ],
        ),
    )

    if dump_graphs:
        logging.info("Edge graph:")
        edge_prog_manager.exported_program().graph_module.graph.print_tabular()

    return edge_prog_manager


def export_to_cadence(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    dump_graphs: bool = False,
    output_dir: Optional[str] = None,
    opt_level: int = 1,
) -> EdgeProgramManager:
    edge_prog_manager = export_to_edge(model, inputs)
    cadence_passes = get_cadence_passes(opt_level)

    # Run a couple required passes for quant/dequant ops
    cadence_prog_manager = edge_prog_manager.transform(
        cast(
            list[Callable[[torch.fx.GraphModule], Optional[PassResult]]], cadence_passes
        )
    )
    return cadence_prog_manager


def quantize_and_export_to_cadence(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    dump_graphs: bool = False,
    opt_level: int = 1,
) -> EdgeProgramManager:
    quantized_model = quantize_pt2(model, inputs)

    return export_to_cadence(
        quantized_model,
        inputs,
        opt_level=opt_level,
        dump_graphs=dump_graphs,
    )


# Export the model and lower it to an EdgeProgramManager (in edge IR), and
# apply passes specific to Cadence DSP execution. Return both to print the
# differences.
def export_to_executorch_gen_etrecord(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    output_dir: Optional[str] = None,
    opt_level: int = 1,
    dump_graphs: bool = False,
) -> ExecutorchProgramManager:
    cadence_passes = get_cadence_passes(opt_level)
    edge_prog_manager = export_to_edge(model, inputs, dump_graphs)

    # Run a couple required passes for quant/dequant ops
    cadence_prog_manager = edge_prog_manager.transform(
        cast(
            list[Callable[[torch.fx.GraphModule], Optional[PassResult]]], cadence_passes
        )
    )

    # Print some information to terminal
    print_ops_info(
        edge_prog_manager.exported_program().graph_module,
        cadence_prog_manager.exported_program().graph_module,
    )

    # Get executorch program after Cadence specific passes
    exec_prog: ExecutorchProgramManager = cadence_prog_manager.to_executorch()
    if output_dir:
        _gen_etrecord(edge_prog_manager, exec_prog, Path(output_dir))
    else:
        logging.warning("No output directory provided, skipping ETRecord generation")

    return exec_prog


def _gen_etrecord(
    edge_program: EdgeProgramManager,
    et_program: ExecutorchProgramManager,
    output_dir: Path,
) -> None:
    etrec_path = output_dir / "etrecord.bin"
    try:
        generate_etrecord(
            et_record=etrec_path,
            edge_dialect_program=edge_program,
            executorch_program=et_program,
        )
        logging.info(f"Generated ETRecord at {etrec_path}")
    except Exception:
        # Any errors here shouldn't block the rest of the flow
        logging.exception("Encountered exception while generating ETRecord")
