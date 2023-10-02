# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import sys

from typing import cast, List, Tuple

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import executorch.exir as exir

import torch
from executorch.backends.qualcomm.passes import qnn_partitioner_passes
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.fx import passes


class SoCModel(enum.Enum):
    SM8450 = 1
    SM8475 = 2
    SM8550 = 3


def qnn_capture_config():
    return exir.CaptureConfig(pt2_mode=True, enable_aot=True, _unlift=True)


def qnn_edge_config() -> exir.EdgeCompileConfig:
    return exir.EdgeCompileConfig(_check_ir_validity=False)


def capture_program(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
) -> exir.ExirExportedProgram:
    edge_prog = exir.capture(
        module,
        inputs,
        qnn_capture_config(),
    ).to_edge(qnn_edge_config())
    return edge_prog.transform(qnn_partitioner_passes)


def draw_graph(title, path, graph_module: torch.fx.GraphModule):
    graph = passes.graph_drawer.FxGraphDrawer(graph_module, title)
    with open(f"{path}/{title}.svg", "wb") as f:
        f.write(graph.get_dot_graph().create_svg())


def get_input_node(node: torch.fx.Node, input_index: int) -> torch.fx.Node:
    return cast(torch.fx.Node, node.args[input_index])


def generate_qnn_executorch_option(
    compiler_specs: List[CompileSpec],
) -> PyQnnManager.QnnExecuTorchHtpBackendOptions:
    option = PyQnnManager.QnnExecuTorchOptionsDefault()
    option.graph_name = "executorch"
    option.htp_options.pd_session = (
        PyQnnManager.QnnExecuTorchHtpPdSession.kHtpUnsignedPd
    )
    option.htp_options.use_conv_hmx = True
    option.htp_options.use_fold_relu = True
    for compiler_spec in compiler_specs:
        if compiler_spec.key == "backend_type":
            option.backend_type = PyQnnManager.QnnExecuTorchBackendType(
                int.from_bytes(compiler_spec.value, sys.byteorder)
            )
        elif compiler_spec.key == "htp_precision":
            option.htp_options.precision = PyQnnManager.QnnExecuTorchHtpPrecision(
                int.from_bytes(compiler_spec.value, sys.byteorder)
            )
        elif compiler_spec.key == "log_level":
            option.log_level = PyQnnManager.QnnExecuTorchLogLevel(
                int.from_bytes(compiler_spec.value, sys.byteorder)
            )
        elif compiler_spec.key == "library_path":
            option.library_path = compiler_spec.value.decode("utf-8")
        elif compiler_spec.key == "htp_performance_mode":
            option.htp_options.performance_mode = (
                PyQnnManager.QnnExecuTorchHtpPerformanceMode(
                    int.from_bytes(compiler_spec.value, sys.byteorder)
                )
            )
        elif compiler_spec.key == "htp_soc_model":
            option.htp_options.soc_model = PyQnnManager.QcomChipset(
                int.from_bytes(compiler_spec.value, sys.byteorder)
            )
        else:
            raise ValueError(f"unknown compiler spec key value: {compiler_spec.key}")

    return option


def generate_qnn_executorch_compiler_spec(
    is_fp16: bool, soc_model: SoCModel, debug: bool, saver: bool = False
) -> List[CompileSpec]:
    """
    Helper function generating compiler specs for Qualcomm AI Engine Direct

    Args:
        is_fp16: If true, the model is compiled to QNN HTP fp16 runtime.
            Note that not all SoC support QNN HTP fp16. Only premium tier SoC
            like Snapdragon 8 Gen 1 or newer can support HTP fp16.
        soc_model: The SoC you plan to run the compiled model. Please check
            SocModel in for supported SoC.
            SM8450 (Snapdragon 8 Gen 1)
            SM8475(Snapdragon 8 Gen 1+)
            SM8550(Snapdragon 8 Gen 2)
        debug: Enable verbose logging. Disclaimer: this option must change in
            the near future.
        saver: Instead of compiling the model, run QNN Saver. Please check
            documents of Qualcomm AI Engine Direct SDK. This feature is usually
            for debugging purpose.

    Returns:
        List[CompileSpec]: Compiler specs for Qualcomm AI Engine Direct.

    Raises:
        ValueError: The value SoCModel is currently not supported.
    """

    supported_soc_models = {
        SoCModel.SM8450: PyQnnManager.QcomChipset.SM8450,
        SoCModel.SM8475: PyQnnManager.QcomChipset.SM8475,
        SoCModel.SM8550: PyQnnManager.QcomChipset.SM8550,
    }
    backend_type = CompileSpec(
        "backend_type", bytes([PyQnnManager.QnnExecuTorchBackendType.kHtpBackend])
    )

    if is_fp16:
        htp_precision = CompileSpec(
            "htp_precision", bytes([PyQnnManager.QnnExecuTorchHtpPrecision.kHtpFp16])
        )
    else:
        htp_precision = CompileSpec(
            "htp_precision",
            bytes([PyQnnManager.QnnExecuTorchHtpPrecision.kHtpQuantized]),
        )

    if debug:
        log_level = CompileSpec(
            "log_level", bytes([PyQnnManager.QnnExecuTorchLogLevel.kLogLevelDebug])
        )
    else:
        log_level = CompileSpec(
            "log_level", bytes([PyQnnManager.QnnExecuTorchLogLevel.kLogLevelWarn])
        )

    # This actually is not an option which can affect the compiled blob.
    # But we don't have other place to pass this option at execution stage.
    htp_performance_mode = CompileSpec(
        "htp_performance_mode",
        bytes([PyQnnManager.QnnExecuTorchHtpPerformanceMode.kHtpBurst]),
    )

    compiler_spec = [backend_type, htp_precision, log_level, htp_performance_mode]

    if soc_model not in supported_soc_models:
        raise ValueError(f"unknown SoC model for QNN: {soc_model}")
    else:
        compiler_spec.append(
            CompileSpec("htp_soc_model", bytes([supported_soc_models[soc_model]]))
        )

    if saver:
        library_path = CompileSpec(
            "library_path", bytes("libQnnSaver.so", encoding="utf-8")
        )
        compiler_spec.append(library_path)

    return compiler_spec
