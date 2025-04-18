# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "generation_config_test",
        srcs = ["generation_config_test.cpp"],
        deps = [
            "//executorch/extension/llm/runner:irunner",
            "//executorch/extension/llm/runner:stats",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ],
    )
