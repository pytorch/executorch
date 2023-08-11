load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    register_custom_op_1 = native.read_config("executorch", "register_custom_op_1", "0") == "1"

    custom_ops_lib = ["//executorch/examples/custom_ops:lib_1"] if register_custom_op_1 else []

    # Test driver for models, uses all portable kernels.
    runtime.cxx_binary(
        name = "executor_runner",
        srcs = ["executor_runner.cpp"],
        deps = [
            "//executorch/runtime/executor/test:test_backend_compiler_lib",
            "//executorch/runtime/executor:program",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/util:util",
            "//executorch/kernels/portable:generated_lib_all_ops",
        ] + custom_ops_lib,
        external_deps = [
            "gflags",
        ],
        define_static_target = True,
        **get_oss_build_kwargs()
    )
