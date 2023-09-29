load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_default_executorch_platforms", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "executorch_generated_lib")
load("@fbsource//xplat/executorch/extension/pybindings:pybindings.bzl", "MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    executorch_generated_lib(
        name = "generated_op_lib_for_runner",
        deps = [
            "//executorch/kernels/optimized:optimized_operators",
            "//executorch/kernels/optimized:optimized_oplist",
            "//executorch/kernels/portable:executorch_aten_ops",
            "//executorch/kernels/portable:executorch_custom_ops",
            "//executorch/kernels/portable:operators",
        ],
        custom_ops_aten_kernel_deps = [
            "//executorch/kernels/portable:operators_aten",
        ],
        functions_yaml_target = "//executorch/kernels/optimized:optimized.yaml",
        custom_ops_yaml_target = "//executorch/kernels/portable:custom_ops.yaml",
        fallback_yaml_target = "//executorch/kernels/portable:functions.yaml",
        define_static_targets = True,
    )

    # Test driver for models, uses one of the following kernels:
    # (1) all portable kernels
    # (2) optimized kernels, with portable kernels as fallback
    for kernel_mode in ["portable", "optimized"]:
        binary_name = "executor_runner" if kernel_mode == "portable" else ("executor_runner_" + kernel_mode)

        runtime.cxx_library(
            name = binary_name + "_lib",
            srcs = ["executor_runner.cpp"],
            exported_deps = [
                "//executorch/runtime/executor/test:test_backend_compiler_lib",
                "//executorch/runtime/executor/test:test_backend_with_delegate_mapping",
                "//executorch/runtime/executor:program",
                "//executorch/sdk/etdump:etdump",
                "//executorch/sdk/etdump:etdump_flatcc",
                "//executorch/util:bundled_program_verification",
                "//executorch/extension/data_loader:buffer_data_loader",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/util:util",
                "//executorch/configurations:executor_cpu_optimized",
                "//executorch/kernels/quantized:generated_lib",
            ] + (MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB if kernel_mode == "portable" else [
                ":generated_op_lib_for_runner",
            ]),
            preprocessor_flags = [],
            external_deps = [
                "gflags",
            ],
            platforms = get_default_executorch_platforms(),
            define_static_target = True,
            visibility = [
                "//executorch/sdk/runners/...",
            ],
        )

        runtime.cxx_binary(
            name = binary_name,
            srcs = [],
            deps = [
                ":" + binary_name + "_lib",
            ],
            external_deps = [
                "gflags",
            ],
            platforms = get_default_executorch_platforms(),
            define_static_target = True,
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
        )

    runtime.export_file(
        name = "executor_runner.cpp",
        visibility = ["//executorch/sdk/runners/..."],
    )
