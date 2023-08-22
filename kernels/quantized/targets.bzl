load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib", "exir_custom_ops_aot_lib")

def define_common_targets():
    runtime.export_file(
        name = "quantized.yaml",
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "all_quantized_ops",
        ops_schema_yaml_target = ":quantized.yaml",
        define_static_targets = True,
    )

    # lib used to register quantized ops into EXIR
    exir_custom_ops_aot_lib(
        name = "aot_lib",
        yaml_target = ":quantized.yaml",
        visibility = ["//executorch/..."],
        kernels = [":quantized_operators_aten"],
        deps = [
            ":all_quantized_ops",
        ],
    )

    for aten_mode in (True, False):
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

        executorch_generated_lib(
            name = "generated_lib" + aten_suffix,
            deps = [
                ":quantized_operators" + aten_suffix,
                ":all_quantized_ops",
            ],
            custom_ops_yaml_target = ":quantized.yaml",
            custom_ops_aten_kernel_deps = [":quantized_operators_aten"] if aten_mode else [],
            aten_mode = aten_mode,
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            define_static_targets = True,
        )
