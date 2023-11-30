load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # executor_runner for MPS Backend and portable kernels.
    if runtime.is_oss:
        runtime.cxx_binary(
            name = "mps_executor_runner",
            srcs = ["mps_executor_runner.mm"],
            deps = [
                "//executorch/backends/apple/mps/runtime:MPSBackend",
                "//executorch/runtime/executor:program",
                "//executorch/extension/evalue_util:print_evalue",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/kernels/portable:generated_lib_all_ops",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/data_loader:buffer_data_loader",
                "//executorch/util:util",
                "//executorch/sdk/bundled_program:runtime",
                "//executorch/util:util",
            ],
            define_static_target = True,
            **get_oss_build_kwargs()
        )
