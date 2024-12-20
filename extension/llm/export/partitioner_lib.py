# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional


def get_xnnpack_partitioner(dynamic_quant_only_partitioner: bool = True):
    """
    Returns the XNNPACK partitioner.

    @arg dynamic_quant_only_partitioner:
        This is enabled by default to keep BC.
        If dynamic_quant_only_partitioner is True, then only dynamically quantized
        linear layers will be partitioned.
        Else, anything which can be will be partitioned greedily.
    """
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
        XnnpackPartitioner,
    )

    if dynamic_quant_only_partitioner:
        # Following changes due to.
        # 1. We need dynamically quantized partitioner for both pt2e_quantize options
        #    as well as "qmode 8da4w" which is also dynamic quantizes linear layers.
        # 2. XNNPACK partitioner seems to result in seg fault for non dqlinear ops.
        return XnnpackDynamicallyQuantizedPartitioner()
    return XnnpackPartitioner()


def get_vulkan_partitioner(
    dtype_override: Optional[str] = None, enable_dynamic_shape: bool = False
):
    assert (
        dtype_override == "fp32" or dtype_override is None
    ), "Vulkan backend does not support non fp32 dtypes at the moment"
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    return VulkanPartitioner({"require_dynamic_shapes": enable_dynamic_shape})


def get_mps_partitioner(use_kv_cache: bool = False):
    from executorch.exir.backend.backend_details import CompileSpec

    assert (
        use_kv_cache is True
    ), "MPS backend currently only supports static shape and use_kv_cache=True is the only way to support it at the moment"
    try:
        # pyre-ignore Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.mps.partition.mps_partitioner`.
        from executorch.backends.apple.mps.partition.mps_partitioner import (
            MPSPartitioner,
        )
    except ImportError:
        raise ImportError(
            "Please install the MPS backend follwing https://pytorch.org/executorch/main/build-run-mps.html"
        )

    compile_specs = [CompileSpec("use_fp16", bytes([True]))]
    return MPSPartitioner(compile_specs)  # pyre-fixme[16]


def get_coreml_partitioner(
    ios: int = 15,
    embedding_quantize: Optional[str] = None,
    pt2e_quantize: Optional[str] = None,
    coreml_quantize: Optional[str] = None,
    coreml_compute_units: Optional[str] = None,
):
    try:
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import (  # pyre-ignore
            CoreMLBackend,
        )
        from executorch.backends.apple.coreml.partition import (  # pyre-ignore
            CoreMLPartitioner,
        )
    except ImportError:
        raise ImportError(
            "Please install the CoreML backend follwing https://pytorch.org/executorch/main/build-run-coreml.html"
        )

    def _validate_ios_version() -> None:
        assert ios in (15, 16, 17, 18)

        if embedding_quantize is not None and ios < 18:
            raise ValueError(
                "In Core ML, per-block quantization is introduced in iOS 18"
            )

        use_quantization = pt2e_quantize is not None or coreml_quantize is not None
        if use_quantization and ios < 16:
            raise ValueError("In Core ML, quantization is introduced in iOS 16")

        use_8a = (pt2e_quantize is not None and "8a" in pt2e_quantize) or (
            coreml_quantize is not None and "8a" in coreml_quantize
        )
        if use_8a and ios < 17:
            raise ValueError(
                "In Core ML, 8-bit activation quantization is introduced in iOS 17"
            )

        use_4w = (pt2e_quantize is not None and "4w" in pt2e_quantize) or (
            coreml_quantize is not None and "4w" in coreml_quantize
        )
        if use_4w and ios < 18:
            raise ValueError(
                "In Core ML, 4-bit weight compression is introduced in iOS 18"
            )

    _validate_ios_version()

    minimum_deployment_target = {
        15: ct.target.iOS15,
        16: ct.target.iOS16,
        17: ct.target.iOS17,
        18: ct.target.iOS18,
    }[ios]

    if coreml_compute_units is None:
        # using `ComputeUnit.ALL` can increase the model load time
        # On iPhone 15 Pro, CPU decode model is over 8x faster than GPU for stories110M,
        # so default to CPU_ONLY
        coreml_compute_units = "cpu_only"
    # pyre-ignore
    coreml_compute_units = {
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
    }[coreml_compute_units.lower()]

    op_linear_quantizer_config = None
    if coreml_quantize == "b4w":
        op_linear_quantizer_config = {
            "mode": "linear_symmetric",
            "dtype": "int4",
            "granularity": "per_block",
            "block_size": 32,
            "weight_threshold": 512,
        }
    elif coreml_quantize == "c4w":
        op_linear_quantizer_config = {
            "mode": "linear_symmetric",
            "dtype": "int4",
            "granularity": "per_channel",
        }

    compile_specs = CoreMLBackend.generate_compile_specs(  # pyre-fixme[16]
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=ct.precision(ct.precision.FLOAT16.value),
        compute_unit=coreml_compute_units,
        model_type=CoreMLBackend.MODEL_TYPE.MODEL,  # pyre-fixme[16]
        op_linear_quantizer_config=op_linear_quantizer_config,
    )

    # ExecuTorch does not build CoreML delegate runtime to handle state
    # when using OSS scripts, so we define take_over_mutable_buffer = False,
    # even when target is iOS18
    # take_over_mutable_buffer = minimum_deployment_target >= ct.target.iOS18
    take_over_mutable_buffer = False
    return CoreMLPartitioner(  # pyre-fixme[16]
        compile_specs=compile_specs,
        take_over_mutable_buffer=take_over_mutable_buffer,
    )


def get_qnn_partitioner(
    use_kv_cache: bool = False,
    pt2e_quantize: Optional[str] = None,
    num_sharding: int = 0,
    soc_model: str = "SM8650",  # default to SM8650
):
    assert (
        use_kv_cache is True
    ), "Qualcomm backend currently only supports static shape and use_kv_cache=True is the only way to support it at the moment"
    try:
        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.partition.qnn_partitioner`
        from executorch.backends.qualcomm.partition.qnn_partitioner import (
            QnnPartitioner,
        )

        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.serialization.qc_schema`
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`
        from executorch.backends.qualcomm.utils.utils import (
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
        )
    except ImportError:
        raise ImportError(
            "Please install the Qualcomm backend following https://pytorch.org/executorch/main/build-run-qualcomm-ai-engine-direct-backend.html"
        )

    use_fp16 = True
    skip_node_op_set = {"llama.fallback.default"}
    if pt2e_quantize is not None:
        use_fp16 = False

    return QnnPartitioner(  # pyre-fixme[16]
        generate_qnn_executorch_compiler_spec(  # pyre-fixme[16]
            soc_model=getattr(QcomChipset, soc_model),  # pyre-fixme[16]
            # pyre-fixme[16]
            backend_options=generate_htp_compiler_spec(
                use_fp16=use_fp16,
                use_multi_contexts=num_sharding > 0,
            ),
            debug=False,
            saver=False,
        ),
        skip_node_id_set={},
        skip_node_op_set=skip_node_op_set,
    )
