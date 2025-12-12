load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "executorch_generated_lib")
load("@fbsource//xplat/executorch/extension/pybindings:pybindings.bzl", "MODELS_ATEN_OPS_ATEN_MODE_GENERATED_LIB", "MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_binary(
        name = "qnn_executor_runner",
        srcs = ["qnn_executor_runner.cpp"],
        deps = [
            "//executorch/runtime/executor/test:test_backend_compiler_lib",
            "//executorch/runtime/executor/test:test_backend_with_delegate_mapping",
            "//executorch/runtime/executor:program",
            "//executorch/devtools/etdump:etdump_flatcc",
            "//executorch/devtools/bundled_program:runtime",
            "//executorch/extension/data_loader:buffer_data_loader",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/runner_util:inputs",
            "//executorch/backends/qualcomm/runtime:runtime",
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/configurations:executor_cpu_optimized",
            "//executorch/extension/llm/custom_ops:custom_ops",
            "//executorch/kernels/quantized:generated_lib",
            "//executorch/kernels/portable:generated_lib",
        ],
        platforms = [ANDROID],
        external_deps = [
            "gflags",
        ],
    )
