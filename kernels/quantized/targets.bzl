load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib", "exir_custom_ops_aot_lib")

def define_common_targets():
    runtime.export_file(
        name = "quantized.yaml",
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Excluding embedding_byte ops because we choose to define them
    # in python separately, mostly to be easy to share with oss.
    et_operator_library(
        name = "quantized_ops_need_aot_registration",
        ops = [
            "quantized_decomposed::add.out",
            "quantized_decomposed::choose_qparams.Tensor_out",
            "quantized_decomposed::choose_qparams_per_token_asymmetric.out",
            "quantized_decomposed::dequantize_per_channel.out",
            "quantized_decomposed::dequantize_per_tensor.out",
            "quantized_decomposed::dequantize_per_tensor.Tensor_out",
            "quantized_decomposed::dequantize_per_token.out",
            "quantized_decomposed::mixed_linear.out",
            "quantized_decomposed::mixed_mm.out",
            "quantized_decomposed::quantize_per_channel.out",
            "quantized_decomposed::quantize_per_tensor.out",
            "quantized_decomposed::quantize_per_tensor.Tensor_out",
            "quantized_decomposed::quantize_per_token.out",
        ],
        define_static_targets = True,
    )

    # lib used to register quantized ops into EXIR
    exir_custom_ops_aot_lib(
        name = "custom_ops_generated_lib",
        yaml_target = ":quantized.yaml",
        visibility = ["//executorch/...", "@EXECUTORCH_CLIENTS"],
        kernels = [":quantized_operators_aten"],
        deps = [
            ":quantized_ops_need_aot_registration",
        ],
    )

    # lib used to register quantized ops into EXIR
    # TODO: merge this with custom_ops_generated_lib
    exir_custom_ops_aot_lib(
        name = "aot_lib",
        yaml_target = ":quantized.yaml",
        visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
        ],
        kernels = [":quantized_operators_aten"],
        deps = [
            ":quantized_ops_need_aot_registration",
        ],
    )

    et_operator_library(
        name = "all_quantized_ops",
        ops_schema_yaml_target = ":quantized.yaml",
        define_static_targets = True,
        visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
        ],
    )

    # On Windows we can only compile these two ops currently, so adding a
    # separate target for this.
    et_operator_library(
        name = "q_dq_ops",
            ops = [
                "quantized_decomposed::dequantize_per_tensor.out",
                "quantized_decomposed::dequantize_per_tensor.Tensor_out",
                "quantized_decomposed::quantize_per_tensor.out",
                "quantized_decomposed::quantize_per_tensor.Tensor_out",
                "quantized_decomposed::dequantize_per_channel.out",
                "quantized_decomposed::quantize_per_channel.out",
            ],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "quantized_operators" + aten_suffix,
            srcs = [],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/kernels/quantized/cpu:quantized_cpu" + aten_suffix,
            ],
        )

        for support_exceptions in [True, False]:
            exception_suffix = "_no_exceptions" if not support_exceptions else ""

            executorch_generated_lib(
                name = "generated_lib" + aten_suffix + exception_suffix,
                deps = [
                    ":quantized_operators" + aten_suffix,
                    ":all_quantized_ops",
                ],
                custom_ops_yaml_target = ":quantized.yaml",
                custom_ops_aten_kernel_deps = [":quantized_operators_aten"] if aten_mode else [],
                custom_ops_requires_aot_registration = False,
                aten_mode = aten_mode,
                support_exceptions = support_exceptions,
                visibility = [
                    "//executorch/...",
                    "@EXECUTORCH_CLIENTS",
                ],
                define_static_targets = True,
            )

            # On Windows we can only compile these two ops currently, so adding a
            # separate target for this.
            executorch_generated_lib(
                name = "q_dq_ops_generated_lib" + aten_suffix + exception_suffix,
                custom_ops_yaml_target = ":quantized.yaml",
                kernel_deps = [
                    "//executorch/kernels/quantized/cpu:op_quantize" + aten_suffix,
                    "//executorch/kernels/quantized/cpu:op_dequantize" + aten_suffix,
                ],
                aten_mode = aten_mode,
                deps = [
                    ":q_dq_ops",
                ],
                support_exceptions = support_exceptions,
                visibility = [
                    "//executorch/...",
                    "@EXECUTORCH_CLIENTS",
                ],
            )

    runtime.python_library(
        name = "quantized_ops_lib",
        srcs = ["__init__.py"],
        deps = [
            "//caffe2:torch",
        ],
        visibility = [
            "//executorch/kernels/quantized/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
