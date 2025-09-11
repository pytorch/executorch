# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")


def define_common_targets():
    runtime.python_library(
        name = "facto_util",
        srcs = [
            "facto_util.py",
        ],
        typing = True,
        deps = [
            "fbcode//caffe2:torch",
            "fbcode//pytorch/facto:facto",
        ],
    )
