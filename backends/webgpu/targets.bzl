# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

_DAWN_DEPS = [
    "fbsource//third-party/dawn:dawn_common",
    "fbsource//third-party/dawn:dawn_native",
    "fbsource//third-party/dawn:dawn_platform",
    "fbsource//third-party/dawn:dawn_proc",
]

def define_common_targets():
    runtime.cxx_library(
        name = "webgpu_backend",
        srcs = native.glob(["runtime/**/*.cpp"]),
        exported_headers = native.glob(["runtime/**/*.h"]),
        compiler_flags = [
            "-DWEBGPU_DAWN_INSTANCE_CAPABILITIES",
            "-fexceptions",
        ],
        link_whole = True,
        exported_deps = _DAWN_DEPS + [
            "//executorch/backends/vulkan/serialization:vk_delegate_schema",
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:named_data_map",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "webgpu_model_loader",
        srcs = ["runner/webgpu_model_loader.cpp"],
        exported_headers = ["runner/webgpu_model_loader.h"],
        compiler_flags = [
            "-DC10_USING_CUSTOM_GENERATED_MACROS",
            "-fexceptions",
        ],
        exported_preprocessor_flags = [
            "-DC10_USING_CUSTOM_GENERATED_MACROS",
        ],
        exported_deps = [
            "//executorch/extension/module:module",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "webgpu_model_loader_test",
        srcs = ["test/native/test_model_loader.cpp"],
        deps = [":webgpu_model_loader"],
    )

    runtime.cxx_test(
        name = "webgpu_utils_test",
        srcs = ["test/native/test_webgpu_utils.cpp"],
        deps = [":webgpu_backend"],
    )

    runtime.cxx_test(
        name = "webgpu_device_header_test",
        srcs = ["test/native/test_webgpu_device_header.cpp"],
        deps = [":webgpu_backend"],
    )

    runtime.cxx_test(
        name = "webgpu_default_context_test",
        srcs = ["test/native/test_webgpu_default_context.cpp"],
        deps = [":webgpu_backend"],
    )

    runtime.cxx_test(
        name = "webgpu_query_pool_test",
        srcs = ["test/native/test_webgpu_query_pool.cpp"],
        deps = [":webgpu_backend"],
    )

    runtime.cxx_test(
        name = "webgpu_query_pool_profiled_test",
        srcs = [
            "runtime/WebGPUQueryPool.cpp",
            "test/native/test_webgpu_query_pool.cpp",
        ],
        compiler_flags = [
            "-DWEBGPU_BACKEND_ENABLE_PROFILING",
            "-DWEBGPU_DAWN_INSTANCE_CAPABILITIES",
            "-fexceptions",
        ],
        deps = [":webgpu_backend"],
    )

    runtime.cxx_test(
        name = "webgpu_execution_options_test",
        srcs = [
            "runtime/WebGPUExecutionOptions.cpp",
            "test/native/test_execution_options.cpp",
        ],
    )
