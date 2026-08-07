# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbcode_macros//build_defs:build_file_migration.bzl", "fbcode_target")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    fbcode_target(
        _kind = runtime.python_test,
        name = "test_webgpu_export_manifest",
        srcs = ["test_webgpu_export_manifest.py"],
        typing = True,
        deps = [
            "//executorch/examples/models/qwen3:webgpu_artifact_manifest",
        ],
    )
