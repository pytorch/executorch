load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "executorch_generated_lib")
load("@fbsource//xplat/executorch/extension/pybindings:pybindings.bzl", "MODELS_ATEN_OPS_ATEN_MODE_GENERATED_LIB", "MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB")

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
    # (2) aten kernels
    # (3) optimized kernels, with portable kernels as fallback
    for kernel_mode in ["portable", "aten", "optimized"]:
        aten_mode = kernel_mode == "aten"
        aten_suffix = ("_aten" if aten_mode else "")
        binary_name = "executor_runner" if kernel_mode == "portable" else ("executor_runner_" + kernel_mode)
        runtime.cxx_binary(
            name = binary_name,
            srcs = ["executor_runner.cpp"],
            deps = [
                "//executorch/runtime/executor/test:test_backend_compiler_lib" + aten_suffix,
                "//executorch/runtime/executor/test:test_backend_with_delegate_mapping" + aten_suffix,
                "//executorch/runtime/executor:program" + aten_suffix,
                "//executorch/sdk/etdump:etdump",
                "//executorch/util:bundled_program_verification" + aten_suffix,
                "//executorch/extension/data_loader:buffer_data_loader",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/util:util" + aten_suffix,
            ] + (MODELS_ATEN_OPS_ATEN_MODE_GENERATED_LIB if aten_mode else [
                "//executorch/configurations:executor_cpu_optimized",
                "//executorch/kernels/quantized:generated_lib",
            ] + (MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB if kernel_mode == "portable" else [
                ":generated_op_lib_for_runner",
            ])),
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            external_deps = [
                "gflags",
            ],
            platforms = [ANDROID, CXX],
            define_static_target = not aten_mode,
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
        )
