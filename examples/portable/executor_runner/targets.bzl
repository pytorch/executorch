load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Wraps a commandline executable that can be linked against any desired
    # kernel or backend implementations. Contains a main() function.
    runtime.cxx_library(
        name = "executor_runner_lib",
        srcs = ["executor_runner.cpp"],
        compiler_flags = ["-Wno-global-constructors"],
        deps = [
            "//executorch/runtime/executor:program",
            "//executorch/devtools/etdump:etdump_flatcc",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/evalue_util:print_evalue",
            "//executorch/extension/runner_util:inputs",
        ],
        external_deps = [
            "gflags",
        ],
        define_static_target = True,
        visibility = [
            "//executorch/examples/...",
        ],
    )

    runtime.cxx_library(
        name = "executor_runner_lib_with_threadpool",
        srcs = ["executor_runner.cpp"],
        compiler_flags = ["-Wno-global-constructors"],
        deps = [
            "//executorch/runtime/executor:program",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/evalue_util:print_evalue",
            "//executorch/extension/runner_util:inputs",
            "//executorch/extension/threadpool:cpuinfo_utils",
            "//executorch/extension/threadpool:threadpool",
        ],
        external_deps = [
            "gflags",
        ],
        define_static_target = True,
        visibility = [
            "//executorch/examples/...",
        ],
    )

    register_custom_op = native.read_config("executorch", "register_custom_op", "0")
    register_quantized_ops = native.read_config("executorch", "register_quantized_ops", "0")

    # Include quantized ops to be able to run quantized model with portable ops
    custom_ops_lib = ["//executorch/kernels/quantized:generated_lib"]
    if register_custom_op == "1":
        custom_ops_lib.append("//executorch/examples/portable/custom_ops:lib_1")
    elif register_custom_op == "2":
        custom_ops_lib.append("//executorch/examples/portable/custom_ops:lib_2")

    # Test driver for models, uses all portable kernels and a demo backend. This
    # is intended to have minimal dependencies. If you want a runner that links
    # against a different backend or kernel library, define a new executable
    # based on :executor_runner_lib.
    runtime.cxx_binary(
        name = "executor_runner",
        srcs = [],
        deps = [
            ":executor_runner_lib",
            "//executorch/runtime/executor/test:test_backend_compiler_lib",
            "//executorch/kernels/portable:generated_lib",
        ] + custom_ops_lib,
        define_static_target = True,
        **get_oss_build_kwargs()
    )

    executorch_generated_lib(
        name = "generated_op_lib_for_runner",
        deps = [
            "//executorch/kernels/optimized:optimized_operators",
            "//executorch/kernels/optimized:optimized_oplist",
            "//executorch/kernels/portable:executorch_aten_ops",
            "//executorch/kernels/portable:executorch_custom_ops",
        ],
        kernel_deps = ["//executorch/kernels/portable:operators",],
        custom_ops_aten_kernel_deps = [
            "//executorch/kernels/portable:operators_aten",
        ],
        functions_yaml_target = "//executorch/kernels/optimized:optimized.yaml",
        custom_ops_yaml_target = "//executorch/kernels/portable:custom_ops.yaml",
        fallback_yaml_target = "//executorch/kernels/portable:functions.yaml",
        define_static_targets = True,
    )

    # Test driver for models, should have all fast CPU kernels
    # (XNNPACK, custom SDPA, etc.) available.
    runtime.cxx_binary(
        name = "executor_runner_opt",
        srcs = [],
        deps = [
            ":executor_runner_lib_with_threadpool",
            ":generated_op_lib_for_runner",
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/configurations:executor_cpu_optimized",
            "//executorch/extension/llm/custom_ops:custom_ops",
            "//executorch/kernels/quantized:generated_lib",
        ],
    )
