# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_vela_tests():
    runtime.export_file(
        name = "vela_external_blocks_test.cpp",
        visibility = ["//executorch/backends/arm/test:vela_external_blocks_test"],
    )

    if runtime.is_oss:
        runtime.cxx_test(
            name = "vela_external_blocks_test",
            srcs = ["vela_external_blocks_test.cpp"],
            compiler_flags = ["-DEXECUTORCH_ARM_VELA_TEST_NO_GLOBAL_INIT"],
            deps = [
                "//executorch/backends/arm/runtime:vela_bin_stream",
                "//executorch/runtime/core:core",
                "//executorch/runtime/core:named_data_map",
                "//executorch/runtime/platform:platform",
            ],
        )
