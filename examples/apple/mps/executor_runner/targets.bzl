#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    if runtime.is_oss:
        # executor_runner for MPS Backend and portable kernels.
        runtime.cxx_binary(
            name = "mps_executor_runner",
            srcs = [
                "mps_executor_runner.mm",
            ],
            deps = [
                "//executorch/backends/apple/mps:mps",
                "//executorch/runtime/executor:program",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/kernels/portable:generated_lib_all_ops",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/data_loader:buffer_data_loader",
                "//executorch/util:util",
                "//executorch/sdk/bundled_program:runtime",
                "//executorch/util:util",
            ],
            external_deps = [
                "gflags",
            ],
            define_static_target = True,
            **get_oss_build_kwargs()
        )
