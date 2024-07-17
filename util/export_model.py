#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
from typing import List, Tuple

import executorch.exir as exir
import torch
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig
from executorch.exir.backend.backend_api import to_backend, validation_disabled
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.tracer import Value
from torch.export._trace import _export

_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)


def export_to_edge(
    model: torch.nn.Module,
    example_inputs: Tuple[Value, ...],
    edge_compile_config=_EDGE_COMPILE_CONFIG,
) -> EdgeProgramManager:
    ep = _export(model, example_inputs, pre_dispatch=True)
    copy.deepcopy(ep)
    ep = torch.export.export(ep.module(), example_inputs)
    copy.deepcopy(ep)
    edge_manager = exir.to_edge(ep, compile_config=edge_compile_config)
    return edge_manager


def export_to_xnnpack(
    model: torch.nn.Module, example_inputs: Tuple[Value, ...], quantize=False
):
    if quantize:
        logging.info("Quantizing Model...")
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

        # TODO(T165162973): This pass shall eventually be folded into quantizer
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_global(operator_config)
        m = torch._export.capture_pre_autograd_graph(model.eval(), example_inputs)
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        model = convert_pt2e(m)

    edge: EdgeProgramManager = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=_EDGE_COMPILE_CONFIG,
    )

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    edge = edge.to_backend(XnnpackPartitioner())

    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_constant_segment=False)
    )
    return exec_prog


def export_to_coreml(
    model: torch.nn.Module, example_inputs: Tuple[Value, ...], compute_units="all"
):
    model = model.eval()
    _CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True, _unlift=False)
    edge = exir.capture(model, example_inputs, _CAPTURE_CONFIG).to_edge(
        _EDGE_COMPILE_CONFIG
    )

    from executorch.backends.apple.coreml.compiler import CoreMLBackend

    lowered_module = to_backend(
        CoreMLBackend.__name__,
        edge.exported_program,
        [CompileSpec("compute_units", bytes(compute_units, "utf-8"))],
    )
    lowered_module(*example_inputs)

    exec_prog = (
        exir.capture(lowered_module, example_inputs, _CAPTURE_CONFIG)
        .to_edge(_EDGE_COMPILE_CONFIG)
        .to_executorch(
            config=exir.ExecutorchBackendConfig(extract_constant_segment=False)
        )
    )
    return exec_prog


def export_to_mps(
    model: torch.nn.Module,
    example_inputs: Tuple[Value, ...],
    use_fp16=True,
    use_partitioner=False,
):
    edge: EdgeProgramManager = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=_EDGE_COMPILE_CONFIG,
    )

    compile_specs = [CompileSpec("use_fp16", bytes([use_fp16]))]

    if use_partitioner:
        from executorch.backends.apple.mps.partition.mps_partitioner import (
            MPSPartitioner,
        )

        edge = edge.to_backend(MPSPartitioner(compile_specs=compile_specs))
        exec_prog = edge.to_executorch(
            config=ExecutorchBackendConfig(extract_constant_segment=False)
        )
    else:
        from executorch.backends.apple.mps.mps_preprocess import MPSBackend

        lowered_module = to_backend(
            MPSBackend.__name__, edge.exported_program(), compile_specs
        )
        exec_prog = (
            exir.capture(
                lowered_module,
                example_inputs,
                exir.CaptureConfig(enable_aot=True, _unlift=False),
            )
            .to_edge(_EDGE_COMPILE_CONFIG)
            .to_executorch(
                config=ExecutorchBackendConfig(extract_constant_segment=False)
            )
        )

    return exec_prog


def export_to_qnn(model: torch.nn.Module, example_inputs: Tuple[Value, ...]):
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.backends.qualcomm.quantizer.quantizer import (
        get_default_8bit_qnn_ptq_config,
        QnnQuantizer,
    )
    from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
        QcomChipset,
    )
    from executorch.backends.qualcomm.utils.utils import (
        capture_program,
        generate_qnn_executorch_compiler_spec,
    )
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = QnnQuantizer()
    quant_config = get_default_8bit_qnn_ptq_config()
    quantizer.set_bit8_op_quant_config(quant_config)

    # Typical pytorch 2.0 quantization flow
    m = torch._export.capture_pre_autograd_graph(model.eval(), example_inputs)
    m = prepare_pt2e(m, quantizer)
    # Calibration
    m(*example_inputs)
    # Get the quantized model
    m = convert_pt2e(m)

    # Capture program for edge IR
    edge_program = capture_program(m, example_inputs)

    # Delegate to QNN backend
    qnn_partitioner = QnnPartitioner(
        generate_qnn_executorch_compiler_spec(
            is_fp16=False,
            soc_model=QcomChipset.SM8550,
            debug=False,
            saver=False,
        )
    )
    with validation_disabled():
        delegated_program = edge_program
        delegated_program.exported_program = to_backend(
            edge_program.exported_program, qnn_partitioner
        )

    executorch_program = delegated_program.to_executorch(
        config=ExecutorchBackendConfig(extract_constant_segment=False)
    )
    return executorch_program


def save_pte_program(buffer: bytes, model_name: str, output_dir: str = "") -> None:
    filename = os.path.join(output_dir, f"{model_name}.pte")
    try:
        with open(filename, "wb") as file:
            file.write(buffer)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")


def export_to_backends(
    model: torch.nn.Module,
    example_inputs: Tuple[Value, ...],
    backends: List[str],
    model_name: str,
    output_dir: str = "",
) -> None:
    """
    Helper function to export to multiple backends
    Available backends (variations): xnnpack_fp32, xnnpack_q8, coreml, mps_fp32, mps_fp16, qnn
    """
    for backend in backends:
        if backend == "xnnpack_fp32":
            exec_prog = export_to_xnnpack(model, example_inputs, False)
        elif backend == "xnnpack_q8":
            exec_prog = export_to_xnnpack(model, example_inputs, True)
        elif backend == "coreml":
            exec_prog = export_to_coreml(model, example_inputs, "all")
        elif backend == "mps_fp32":
            exec_prog = export_to_mps(model, example_inputs, False, False)
        elif backend == "mps_fp16":
            exec_prog = export_to_mps(model, example_inputs, True, False)
        elif backend == "qnn":
            exec_prog = export_to_qnn(model, example_inputs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        save_pte_program(exec_prog.buffer, model_name + "_" + backend, output_dir)
