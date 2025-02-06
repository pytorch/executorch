# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")


def define_common_targets():
    python_library(
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
