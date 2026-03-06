# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.python_library(
        name = "graph_builder",
        srcs = [
            "graph_builder.py",
        ],
        typing = True,
        deps = [
            "//caffe2:torch",
            "//executorch/exir:pass_base",
        ],
    )

    runtime.python_library(
        name = "program_builder",
        srcs = [
            "program_builder.py",
        ],
        typing = True,
        deps = [
            ":graph_builder",
            "//caffe2:torch",
            "//executorch/exir:lib",
            "//executorch/exir:pass_base",
            "//executorch/exir/verification:verifier",
        ],
    )

    if not runtime.is_oss and is_fbcode:
        modules_env = {
           "ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH": "$(location fbcode//executorch/test/models:exported_xnnp_delegated_programs[ModuleAddLarge.pte])",
           "ET_XNNPACK_GENERATED_SUB_LARGE_PTE_PATH": "$(location fbcode//executorch/test/models:exported_xnnp_delegated_programs[ModuleSubLarge.pte])",
        }

        runtime.cxx_test(
            name = "multi_method_delegate_test",
            srcs = [
                "multi_method_delegate_test.cpp",
            ],
            deps = [
                "//executorch/runtime/executor:program",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/memory_allocator:malloc_memory_allocator",
                "//executorch/kernels/portable:generated_lib",
                "//executorch/backends/xnnpack:xnnpack_backend",
                "//executorch/extension/runner_util:inputs",
            ],
            env = modules_env,
        )
