load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    register_custom_op = native.read_config("executorch", "register_custom_op", "0")

    if register_custom_op == "1":
        custom_ops_lib = ["//executorch/examples/custom_ops:lib_1"]
    elif register_custom_op == "2":
        custom_ops_lib = ["//executorch/examples/custom_ops:lib_2"]
    else:
        custom_ops_lib = []

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
