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
from executorch.backends.qualcomm.passes.convert_hardsigmoid import ConvertHardsigmoid
from executorch.backends.qualcomm.passes.convert_interpolate_with_upsample2d import (
    ConvertInterpolateWithUpsample2D,
)
from executorch.backends.qualcomm.passes.convert_to_linear import ConvertToLinear
from executorch.backends.qualcomm.passes.fold_qdq import FoldQDQ
from executorch.backends.qualcomm.passes.i64_to_i32 import I64toI32
from executorch.backends.qualcomm.passes.layout_transform import LayoutTransform
from executorch.backends.qualcomm.passes.remove_clone import RemoveClone
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    _soc_info_table,
    QcomChipset,
    QnnExecuTorchBackendOptions,
    QnnExecuTorchBackendType,
    QnnExecuTorchHtpBackendOptions,
    QnnExecuTorchHtpPerformanceMode,
    QnnExecuTorchHtpPrecision,
    QnnExecuTorchLogLevel,
    QnnExecuTorchOptions,
    QnnExecuTorchProfileLevel,
)
from executorch.backends.qualcomm.serialization.qnn_compile_spec_serialize import (
    convert_to_flatbuffer,
    convert_to_option,
)
from executorch.exir import ExirExportedProgram
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._decomp import core_aten_decompositions as torch_core_aten_decompositions
from torch.export.exported_program import ExportedProgram
from torch.fx import passes

QNN_COMPILE_SPEC = "qnn_compile_spec"


def qnn_capture_config():
    return exir.CaptureConfig(enable_aot=True)


def qnn_edge_config() -> exir.EdgeCompileConfig:
    return exir.EdgeCompileConfig(_check_ir_validity=False)


def canonicalize_program(prog: ExportedProgram):
    # check if user specifies to use multi_contexts
    # this is a generic approach in case there exists multiple backends
    max_sf_buf_size, modules = 0, {}
    for _, m in prog.graph_module._modules.items():
        # currently only 1 compile spec is expected in each partition
        options = convert_to_option(m.compile_specs[0].value)
        if (
            options.backend_options.backend_type == QnnExecuTorchBackendType.kHtpBackend
            and options.backend_options.htp_options.use_multi_contexts
        ):
            max_sf_buf_size = max(max_sf_buf_size, len(m.processed_bytes))
            modules[m] = options

    if max_sf_buf_size != 0:
        for module, options in modules.items():
            options.backend_options.htp_options.max_sf_buf_size = max_sf_buf_size
            module.compile_specs[0] = CompileSpec(
                QNN_COMPILE_SPEC, convert_to_flatbuffer(options)
            )


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


def _transform(edge_program: ExportedProgram) -> None:
    # currently ExirExportedProgram.transform does not accept
    # changes of input number which was caused by FoldQDQ
    # apply passes one by one here to avoid IR capture failure
    graph_module = edge_program.graph_module
    RemoveClone()(graph_module)
    ConvertToLinear()(graph_module)
    ConvertHardsigmoid()(graph_module)
    ConvertBmmToMatmul()(graph_module)
    ConvertInterpolateWithUpsample2D()(graph_module)
    I64toI32(edge_program)(graph_module)
    AnnotateQuantAttrs(edge_program)(graph_module)
    AnnotateAndQuantScalar(edge_program)(graph_module)
    AnnotateDecomposed(edge_program)(graph_module)
    FoldQDQ()(graph_module)
    LayoutTransform(edge_program)(graph_module)


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
    _transform(edge_ep.exported_program)
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


def generate_htp_compiler_spec(
    use_fp16: bool,
    use_dlbc: bool = False,
    use_multi_contexts: bool = False,
) -> QnnExecuTorchBackendOptions:
    """
    Helper function generating backend options for QNN HTP

    Args:
        use_fp16: If true, the model is compiled to QNN HTP fp16 runtime.
            Note that not all SoC support QNN HTP fp16. Only premium tier SoC
            like Snapdragon 8 Gen 1 or newer can support HTP fp16.
        use_dlbc: Deep Learning Bandwidth Compression allows inputs to be
            compressed, such that the processing bandwidth can be lowered.
        use_multi_contexts: When multiple contexts are generated inside the same
            pte, it is possible to reserve a single spill-fill allocation that
            could be re-used across all the splits.

    Returns:
        QnnExecuTorchHtpBackendOptions: backend options for QNN HTP.
    """
    htp_options = QnnExecuTorchHtpBackendOptions()
    htp_options.precision = (
        QnnExecuTorchHtpPrecision.kHtpFp16
        if use_fp16
        else QnnExecuTorchHtpPrecision.kHtpQuantized
    )
    # This actually is not an option which can affect the compiled blob.
    # But we don't have other place to pass this option at execution stage.
    # TODO: enable voting mechanism in runtime and make this as an option
    htp_options.performance_mode = QnnExecuTorchHtpPerformanceMode.kHtpBurst
    htp_options.use_multi_contexts = use_multi_contexts
    htp_options.use_dlbc = use_dlbc
    return QnnExecuTorchBackendOptions(
        backend_type=QnnExecuTorchBackendType.kHtpBackend,
        htp_options=htp_options,
    )


def generate_qnn_executorch_compiler_spec(
    soc_model: QcomChipset,
    backend_options: QnnExecuTorchBackendOptions,
    debug: bool = False,
    saver: bool = False,
    online_prepare: bool = False,
    tensor_dump_output_path: str = "",
    profile: bool = False,
    shared_buffer: bool = False,
) -> List[CompileSpec]:
    """
    Helper function generating compiler specs for Qualcomm AI Engine Direct

    Args:
        soc_model: The SoC you plan to run the compiled model. Please check
            QcomChipset for supported SoC.
            SM8450 (Snapdragon 8 Gen 1)
            SM8475(Snapdragon 8 Gen 1+)
            SM8550(Snapdragon 8 Gen 2)
            SM8650(Snapdragon 8 Gen 3)
        backend_options: Options required by different backends.
        debug: Enable verbose logging. Disclaimer: this option must change in
            the near future.
        online_prepare: Compose QNN graph on device if set to True
        saver: Instead of compiling the model, run QNN Saver. Please check
            documents of Qualcomm AI Engine Direct SDK. This feature is usually
            for debugging purpose.
        tensor_dump_output_path: If a path is given, Delegate would write
            outputs of each OP there in runtime. In ALL cases,
            we don't recommend to set this option. This option exist just
            for debugging some accuracy issues.
        profile: Enable profile the performance of per operator.
            Note that for now only support kProfileDetailed to
            profile the performance of each operator with cycle unit.
        shared_buffer: Enables usage of shared buffer between application
            and backend for graph I/O.

    Returns:
        List[CompileSpec]: Compiler specs for Qualcomm AI Engine Direct.

    Raises:
        ValueError: The value QcomChipset is currently not supported.
        ValueError: Confliction between compiler specs.
    """
    _supported_soc_models = {soc_model.value for soc_model in QcomChipset}
    if soc_model not in _supported_soc_models:
        raise ValueError(f"unknown SoC model for QNN: {soc_model}")

    qnn_executorch_options = QnnExecuTorchOptions(
        _soc_info_table[soc_model], backend_options
    )
    qnn_executorch_options.graph_name = "executorch"
    qnn_executorch_options.log_level = (
        QnnExecuTorchLogLevel.kLogLevelDebug
        if debug
        else QnnExecuTorchLogLevel.kLogLevelWarn
    )

    if saver:
        qnn_executorch_options.library_path = "libQnnSaver.so"

    if len(tensor_dump_output_path.strip()) != 0:
        qnn_executorch_options.tensor_dump_output_path = tensor_dump_output_path

    if profile:
        qnn_executorch_options.profile_level = (
            QnnExecuTorchProfileLevel.kProfileDetailed
        )
    else:
        qnn_executorch_options.profile_level = QnnExecuTorchProfileLevel.kProfileOff

    if shared_buffer:
        qnn_executorch_options.shared_buffer = True

    if (
        online_prepare
        and backend_options.backend_type == QnnExecuTorchBackendType.kHtpBackend
        and backend_options.htp_options.use_multi_contexts
    ):
        raise ValueError(
            "'use_multi_context' could not function in online prepare mode, "
            "please set 'online_prepare' to False"
        )

    qnn_executorch_options.online_prepare = online_prepare

    return [
        CompileSpec(QNN_COMPILE_SPEC, convert_to_flatbuffer(qnn_executorch_options))
    ]
