# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "test_generation_config",
        srcs = ["test_generation_config.cpp"],
        deps = [
            "//executorch/extension/llm/runner:irunner",
            "//executorch/extension/llm/runner:stats",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )

    runtime.cxx_test(
        name = "test_text_llm_runner",
        srcs = ["test_text_llm_runner.cpp"],
        deps = [
            "//executorch/extension/llm/runner:irunner",
            "//executorch/extension/llm/runner:runner_lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "test_text_prefiller",
        srcs = ["test_text_prefiller.cpp"],
        deps = [
            "//executorch/extension/llm/runner:runner_lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )

    runtime.cxx_test(
        name = "test_text_decoder_runner",
        srcs = ["test_text_decoder_runner.cpp"],
        deps = [
            "//executorch/extension/llm/runner:runner_lib",
            "//executorch/kernels/portable:generated_lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
        env = {
            "KVCACHE_CACHE_POS": "$(location fbcode//executorch/test/models:exported_programs[ModuleKVCacheCachePos.pte])",
            "KVCACHE_INPUT_POS": "$(location fbcode//executorch/test/models:exported_programs[ModuleKVCacheInputPos.pte])",
            "NO_KVCACHE": "$(location fbcode//executorch/test/models:exported_programs[ModuleNoKVCache.pte])",
        }
    )
