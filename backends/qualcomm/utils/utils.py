# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Tuple

import executorch.exir as exir

import torch

from executorch.backends.qualcomm.passes.annotate_and_quant_scalar import (
    AnnotateAndQuantScalar,
)
from executorch.backends.qualcomm.passes.annotate_decomposed import AnnotateDecomposed
from executorch.backends.qualcomm.passes.annotate_quant_attrs import AnnotateQuantAttrs
from executorch.backends.qualcomm.passes.convert_binary_op_with_scalar import (
    ConvertBinaryOpsWithScalar,
)
from executorch.backends.qualcomm.passes.convert_bmm_to_matmul import ConvertBmmToMatmul
from executorch.backends.qualcomm.passes.convert_interpolate_with_upsample2d import (
    ConvertInterpolateWithUpsample2D,
)
from executorch.backends.qualcomm.passes.convert_to_linear import ConvertToLinear
from executorch.backends.qualcomm.passes.fold_qdq import FoldQDQ
from executorch.backends.qualcomm.passes.i64_to_i32 import I64toI32
from executorch.backends.qualcomm.passes.insert_requantize import InsertRequantize
from executorch.backends.qualcomm.passes.layout_transform import LayoutTransform
from executorch.backends.qualcomm.passes.remove_clone import RemoveClone
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    _soc_info_table,
    QcomChipset,
    QnnExecuTorchBackendType,
    QnnExecuTorchHtpPdSession,
    QnnExecuTorchHtpPerformanceMode,
    QnnExecuTorchHtpPrecision,
    QnnExecuTorchLogLevel,
    QnnExecuTorchOptions,
)
from executorch.backends.qualcomm.serialization.qnn_compile_spec_serialize import (
    convert_to_flatbuffer,
)
from executorch.exir import ExirExportedProgram
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._decomp import core_aten_decompositions as torch_core_aten_decompositions
from torch.fx import passes

QNN_COMPILE_SPEC = "qnn_compile_spec"


def qnn_capture_config():
    return exir.CaptureConfig(enable_aot=True)


def qnn_edge_config() -> exir.EdgeCompileConfig:
    return exir.EdgeCompileConfig(_check_ir_validity=False)


def get_decomp_table() -> Dict[torch._ops.OperatorBase, Callable]:
    source_decompositions = torch_core_aten_decompositions()
    # The below super ops are supported by QNN
    remove_decompositions = [
        torch.ops.aten.pixel_shuffle.default,
        torch.ops.aten.hardswish.default,
    ]

    for key in remove_decompositions:
        source_decompositions.pop(key)

    return source_decompositions


def capture_program(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
) -> exir.ExirExportedProgram:
    ep = torch.export.export(module, inputs)
    decomposed_ep = ep.run_decompositions(get_decomp_table())

    # We choose call_operator by target in ConvertBinaryOpsWithScalar
    # because it is the same source_fn_stack for MultiheadAttention
    # TODO: Should modify the scalar op in the op builder instead of
    #       using transformation
    core_ep = ExirExportedProgram(decomposed_ep, False)
    core_ep.transform(ConvertBinaryOpsWithScalar())
    edge_ep = core_ep.to_edge(qnn_edge_config())

    # currently ExirExportedProgram.transform does not accept
    # changes of input number which was caused by FoldQDQ
    # apply passes one by one here to avoid IR capture failure
    edge_program = edge_ep.exported_program
    graph_module = edge_program.graph_module
    RemoveClone()(graph_module)
    ConvertToLinear()(graph_module)
    ConvertBmmToMatmul()(graph_module)
    ConvertInterpolateWithUpsample2D()(graph_module)
    I64toI32(edge_program)(graph_module)
    AnnotateQuantAttrs(edge_program)(graph_module)
    AnnotateAndQuantScalar(edge_program)(graph_module)
    AnnotateDecomposed(edge_program)(graph_module)
    FoldQDQ()(graph_module)
    InsertRequantize(edge_program)(graph_module)
    LayoutTransform(edge_program)(graph_module)
    return edge_ep


def draw_graph(title, path, graph_module: torch.fx.GraphModule):
    graph = passes.graph_drawer.FxGraphDrawer(graph_module, title)
    with open(f"{path}/{title}.svg", "wb") as f:
        f.write(graph.get_dot_graph().create_svg())


def generate_qnn_executorch_option(
    compiler_specs: List[CompileSpec],
) -> bytes:
    for compiler_spec in compiler_specs:
        if compiler_spec.key == QNN_COMPILE_SPEC:
            qnn_compile_spec_buffer = compiler_spec.value
        else:
            raise ValueError(f"unknown compiler spec key value: {compiler_spec.key}")
    return qnn_compile_spec_buffer


# TODO: refactor this for supporting other backends
def generate_qnn_executorch_compiler_spec(
    is_fp16: bool,
    soc_model: QcomChipset,
    debug: bool = False,
    saver: bool = False,
    online_prepare: bool = False,
) -> List[CompileSpec]:
    """
    Helper function generating compiler specs for Qualcomm AI Engine Direct

    Args:
        is_fp16: If true, the model is compiled to QNN HTP fp16 runtime.
            Note that not all SoC support QNN HTP fp16. Only premium tier SoC
            like Snapdragon 8 Gen 1 or newer can support HTP fp16.
        soc_model: The SoC you plan to run the compiled model. Please check
            QcomChipset for supported SoC.
            SM8450 (Snapdragon 8 Gen 1)
            SM8475(Snapdragon 8 Gen 1+)
            SM8550(Snapdragon 8 Gen 2)
            SM8650(Snapdragon 8 Gen 3)
        online_prepare: Compose QNN graph on device if set to True
        debug: Enable verbose logging. Disclaimer: this option must change in
            the near future.
        saver: Instead of compiling the model, run QNN Saver. Please check
            documents of Qualcomm AI Engine Direct SDK. This feature is usually
            for debugging purpose.

    Returns:
        List[CompileSpec]: Compiler specs for Qualcomm AI Engine Direct.

    Raises:
        ValueError: The value QcomChipset is currently not supported.
    """
    qnn_executorch_options = QnnExecuTorchOptions()
    qnn_executorch_options.backend_type = QnnExecuTorchBackendType.kHtpBackend
    qnn_executorch_options.graph_name = "executorch"
    qnn_executorch_options.htp_options.pd_session = (
        QnnExecuTorchHtpPdSession.kHtpUnsignedPd
    )
    qnn_executorch_options.htp_options.use_conv_hmx = True
    qnn_executorch_options.htp_options.use_fold_relu = True

    if is_fp16:
        qnn_executorch_options.htp_options.precision = (
            QnnExecuTorchHtpPrecision.kHtpFp16
        )
    else:
        qnn_executorch_options.htp_options.precision = (
            QnnExecuTorchHtpPrecision.kHtpQuantized
        )

    if debug:
        qnn_executorch_options.log_level = QnnExecuTorchLogLevel.kLogLevelDebug
    else:
        qnn_executorch_options.log_level = QnnExecuTorchLogLevel.kLogLevelWarn

    # This actually is not an option which can affect the compiled blob.
    # But we don't have other place to pass this option at execution stage.
    qnn_executorch_options.htp_options.performance_mode = (
        QnnExecuTorchHtpPerformanceMode.kHtpBurst
    )

    _supported_soc_models = {soc_model.value for soc_model in QcomChipset}
    if soc_model not in _supported_soc_models:
        raise ValueError(f"unknown SoC model for QNN: {soc_model}")
    else:
        qnn_executorch_options.soc_info = _soc_info_table[soc_model]

    if saver:
        qnn_executorch_options.library_path = "libQnnSaver.so"

    if online_prepare:
        qnn_executorch_options.online_prepare = True
    return [
        CompileSpec(QNN_COMPILE_SPEC, convert_to_flatbuffer(qnn_executorch_options))
    ]
