# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.python_library(
        name = "webgpu_artifact_manifest",
        srcs = ["webgpu_artifact_manifest.py"],
        base_module = "executorch.backends.webgpu.scripts",
        visibility = ["PUBLIC"],
    )

    runtime.python_binary(
        name = "webgpu_artifact_manifest_cli",
        main_module = "executorch.backends.webgpu.scripts.webgpu_artifact_manifest",
        deps = [":webgpu_artifact_manifest"],
        visibility = ["PUBLIC"],
    )
