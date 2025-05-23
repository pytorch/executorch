# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "test_runner",
        srcs = ["test_runner.cpp"],
        deps = [
            "//executorch/examples/models/llama/runner:runner",
            "//executorch/extension/llm/runner:irunner",
            "//executorch/extension/llm/runner:stats",
            "//executorch/extension/llm/runner:text_token_generator",
            "//executorch/extension/llm/runner:text_decoder_runner",
            "//executorch/extension/llm/runner:text_prefiller",
            "//executorch/extension/module:module",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )
