load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Wraps a commandline executable that can be linked against any desired
    # kernel or backend implementations. Contains a main() function.
    select_ops = native.read_config("executorch", "select_ops", None)

    runtime.cxx_library(
        name = "executor_runner_lib",
        srcs = ["executor_runner.cpp"],
        deps = [
            "//executorch/runtime/executor:program",
            "//executorch/runtime/kernel:operator_registry",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/evalue_util:print_evalue",
            "//executorch/util:util",
        ] + (["//executorch/examples/selective_build:select_add_out_manual_lib"] if select_ops == "add_manual" else []),
        preprocessor_flags = ["-DEXECUTORCH_MANUAL_KERNEL_REG"] if select_ops == "add_manual" else [],
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
        custom_ops_lib.append("//executorch/examples/custom_ops:lib_1")
    elif register_custom_op == "2":
        custom_ops_lib.append("//executorch/examples/custom_ops:lib_2")

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
            "//executorch/kernels/portable:generated_lib_all_ops",
        ] + custom_ops_lib,
        define_static_target = True,
        **get_oss_build_kwargs()
    )
