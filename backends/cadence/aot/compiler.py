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
from executorch.backends.cadence.aot.memory_planning import (
    CadenceMemoryPlanning,
    print_memory_planning_info,
)
from executorch.backends.cadence.aot.quantizer.fusion_pass import QuantFusion
from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceDefaultQuantizer,
    CadenceQuantizer,
)
from executorch.backends.cadence.aot.utils import (
    get_default_memory_config,
    MemoryConfig,
)
from executorch.devtools import generate_etrecord
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
)
from executorch.exir.pass_base import PassResult
from executorch.exir.passes import ToOutVarPass
from executorch.exir.passes.sym_shape_eval_pass import HintBasedSymShapeEvalPass
from executorch.exir.program._program import to_edge_with_preserved_ops
from torch._inductor.decomposition import remove_decompositions

from torch.export.exported_program import ExportedProgram
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from .passes import get_cadence_passes

from .utils import print_ops_info


default_quantizer = CadenceDefaultQuantizer()


# Note: this is not meant as a primary API since it can create inconsistencies
# if the quantizer here is different from the quantizer used to convert. It is
# however useful for unit tests to separate the converted model from the fused
# model, to be able to get reference numerics.
# If this does not apply, please use quantize_and_fuse_pt2 instead.
def trace(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    dump_graphs: bool = False,
) -> ExportedProgram:
    """
    Trace the model with export_for_training and return an ExportedProgram.
    """

    # Make the model inference mode by calling model.eval()
    model.eval()

    # Prevent mkldnn decompositions
    torch._C._set_mkldnn_enabled(False)

    # Get default decompositions
    decomp_table = torch.export.default_decompositions()

    # Select ops to keep
    ops_to_keep = [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.rms_norm.default,
    ]

    # Remove decompositions for the ops we want to keep
    # pyre-fixme[6]: For 1st argument expected `Dict[typing.Callable[..., typing.Any
    remove_decompositions(decomp_table, ops_to_keep)

    # Export with dynamo
    program = torch.export.export_for_training(
        model, inputs, strict=True
    ).run_decompositions(decomp_table)

    if dump_graphs:
        logging.info("Graph before quantization:")
        logging.info(program.module().graph.print_tabular())

    return program


def prepare_and_convert_pt2(
    program: ExportedProgram,
    inputs: tuple[object, ...],
    quantizer: CadenceQuantizer,
    calibration_data: Optional[list[tuple[object, ...]]] = None,
    dump_graphs: bool = False,
) -> torch.fx.GraphModule:
    """
    Prepare and convert a model using the given quantizer.
    The quantizer must be supplied and be the same as the one used to
    fuse the model later, if applicable. If you do not expect that behavior,
    please use quantize_and_fuse_pt2 instead, which will instantiate a
    default quantizer for you if needed.
    If calibration data is provided, it will be used to calibrate the model. If
    not, the inputs will be used for calibration instead, which is useful for
    unit tests but should not be used for end-to-end use cases.
    Returns a GraphModule with the converted model.
    """

    # Get the graph module from the ExportedProgram
    model_gm = program.module()

    assert isinstance(model_gm, torch.fx.GraphModule)

    # Prepare
    prepared_model = prepare_pt2e(model_gm, quantizer)

    # Calibrate
    # If no calibration data is provided, use the inputs
    if calibration_data is None:
        calibration_data = [inputs]

    for samples in calibration_data:
        prepared_model(*samples)

    # Convert
    converted_model = convert_pt2e(prepared_model)

    if dump_graphs:
        logging.info("Graph after quantization (before fusion):")
        logging.info(model_gm.graph.print_tabular())

    return converted_model


# Note: this is not meant as a primary API since it can create inconsistencies
# if the quantizer here is different from the quantizer used to prepare/convert.
# It is however useful for unit tests to separate the converted model from the
# fused model, to be able to get reference numerics.
# If this does not apply, please use quantize_pt2 instead.
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


def quantize_pt2(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    quantizer: Optional[CadenceQuantizer] = None,
    calibration_data: Optional[list[tuple[object, ...]]] = None,
    dump_graphs: bool = False,
) -> ExportedProgram:
    """
    Trace, prepare, convert and fuse the model using the given quantizer.
    If calibration data is provided, it will be used to calibrate the model. If
    not, the inputs will be used for calibration instead, which is useful for
    unit tests but should not be used for end-to-end use cases.
    Returns a GraphModule with the quantized model.
    Note: this function should not be called directly in general. Please use
    quantize_and_export_to_executorch for most needs.
    """
    # Make the model inference mode by calling model.eval()
    model.eval()

    # Instantiate the quantizer to CadenceQuantizer if not supplied
    if not quantizer:
        quantizer = CadenceDefaultQuantizer()

    program = trace(model, inputs, dump_graphs=dump_graphs)

    if dump_graphs:
        logging.info("Graph after trace:")
        logging.info(program.graph.print_tabular())

    # Get converted graph module
    converted_gm = prepare_and_convert_pt2(
        program, inputs, quantizer, calibration_data, dump_graphs=dump_graphs
    )

    # Get fused model
    fused_gm = fuse_pt2(converted_gm, quantizer)

    if dump_graphs:
        logging.info("Graph after quantization and fusion:")
        logging.info(fused_gm.graph.print_tabular())

    program = torch.export.export(fused_gm, inputs, strict=True)

    return program


def _lower_ep_to_edge(
    expo_program: ExportedProgram,
    dump_graphs: bool = False,
    constant_methods: Optional[dict[str, object]] = None,
) -> EdgeProgramManager:
    """
    Lower an ExportedProgram to an EdgeProgramManager (in edge IR).
    """
    # Call to_edge_with_preserved_ops to convert the graph to edge IR.
    # Note: dim_order is skipped (https://github.com/pytorch/executorch/issues/3704)
    edge_prog_manager = to_edge_with_preserved_ops(
        expo_program,
        compile_config=EdgeCompileConfig(
            _skip_dim_order=True,
            # Allow specific non-core aten ops in the IR.
            _core_aten_ops_exception_list=[
                torch.ops.aten._linalg_det.default,
                torch.ops.aten._linalg_svd.default,
                torch.ops.aten._native_batch_norm_legit_functional.default,
                torch.ops.aten.linear.default,
                torch.ops.aten.linalg_vector_norm.default,
                torch.ops.aten.unfold.default,
                torch.ops.aten.angle.default,
                torch.ops.aten.rms_norm.default,
            ],
        ),
        constant_methods=constant_methods,
        preserve_ops=(torch.ops.aten.rms_norm.default,),
    )

    if dump_graphs:
        logging.info("Graph after Edge lowering:")
        logging.info(
            edge_prog_manager.exported_program().graph_module.graph.print_tabular()
        )
    return edge_prog_manager


# Export the model and lower it to an EdgeProgramManager (in edge IR).
def export_to_edge(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    dump_graphs: bool = False,
    constant_methods: Optional[dict[str, object]] = None,
) -> EdgeProgramManager:
    assert isinstance(model, torch.nn.Module), "model should be an nn.Module"

    # Export the model into an ExportedProgram.
    expo_program = trace(model, inputs)

    # Lower the model to edge IR.
    edge_prog_manager = _lower_ep_to_edge(expo_program, dump_graphs, constant_methods)

    return edge_prog_manager


def quantize_and_export_to_edge(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    quantizer: Optional[CadenceQuantizer] = None,
    dump_graphs: bool = False,
    constant_methods: Optional[dict[str, object]] = None,
    calibration_data: Optional[list[tuple[object, ...]]] = None,
) -> EdgeProgramManager:
    """
    Trace, quantize and lower a model/inputs pair to edge IR.
    """
    quantized_model = quantize_pt2(
        model,
        inputs,
        quantizer=quantizer,
        calibration_data=calibration_data,
        dump_graphs=dump_graphs,
    )

    return _lower_ep_to_edge(
        quantized_model,
        dump_graphs=dump_graphs,
        constant_methods=constant_methods,
    )


def _lower_ep_to_cadence(
    program: ExportedProgram,
    dump_graphs: bool = False,
    opt_level: int = 1,
) -> EdgeProgramManager:
    """
    Lower an existing ExportedProgram to edge IR and apply frontend optimization passes.
    """
    edge_prog_manager = _lower_ep_to_edge(program, dump_graphs=dump_graphs)
    cadence_passes = get_cadence_passes(opt_level)

    # Run a couple required passes for quant/dequant ops
    cadence_prog_manager = edge_prog_manager.transform(
        cast(
            list[Callable[[torch.fx.GraphModule], Optional[PassResult]]], cadence_passes
        )
    )
    return cadence_prog_manager


def export_to_cadence(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    dump_graphs: bool = False,
    opt_level: int = 1,
) -> EdgeProgramManager:
    edge_prog_manager = export_to_edge(model, inputs, dump_graphs=dump_graphs)
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
    """
    Trace, quantize, lower a model/inputs pair to edge IR and apply frontend
    optimization passes.
    """
    quantized_model = quantize_pt2(model, inputs)

    return _lower_ep_to_cadence(
        quantized_model,
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
    mem_algo: int = 0,
    alloc_graph_input: bool = True,
    alloc_graph_output: bool = True,
    memory_config: Optional[MemoryConfig] = None,
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

    if memory_config is None:
        memory_config = get_default_memory_config()

    memory_planning_pass = CadenceMemoryPlanning(
        memory_config,
        opt_level=opt_level,
        mem_algo=mem_algo,
        alloc_graph_input=alloc_graph_input,
        alloc_graph_output=alloc_graph_output,
    )

    # Get executorch program after Cadence specific passes
    exec_prog: ExecutorchProgramManager = cadence_prog_manager.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=memory_planning_pass,
            emit_stacktrace=False,
            to_out_var_pass=ToOutVarPass(),
            extract_delegate_segments=False,
            sym_shape_eval_pass=HintBasedSymShapeEvalPass(),
        ),
    )

    print_memory_planning_info(
        exec_prog,
        memory_config,
        opt_level,
        alloc_graph_input,
        alloc_graph_output,
    )

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
