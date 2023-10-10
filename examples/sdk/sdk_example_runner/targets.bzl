load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Test driver for models with bundled inputs.
    runtime.cxx_binary(
        name = "sdk_example_runner",
        srcs = [
            "sdk_example_runner.cpp",
        ],
        deps = [
            "//executorch/runtime/executor/test:test_backend_compiler_lib",
            "//executorch/kernels/portable:generated_lib_all_ops",
            "//executorch/runtime/executor:program",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/data_loader:buffer_data_loader",
            "//executorch/util:util",
            "//executorch/util:bundled_program_verification",
            "//executorch/sdk/etdump:etdump_flatcc",
        ],
        external_deps = [
            "gflags",
        ],
        define_static_target = True,
        **get_oss_build_kwargs()
    )
