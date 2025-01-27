# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

TESTS_LIST = [
    "add_op",
    "g3_ops",
    "quantized_conv1d_op",
    "quantized_linear_op",
]

def define_common_targets():
    for op in TESTS_LIST:
        _define_test_target(op)

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


def _define_test_target(test_name):
    file_name = "test_{}".format(test_name)
    python_unittest(
        name = file_name,
        srcs = [
            "{}.py".format(file_name),
        ],
        typing = True,
        supports_static_listing = False,
        deps = [
            "fbsource//third-party/pypi/parameterized:parameterized",
            "fbcode//caffe2:torch",
            "fbcode//executorch/backends/cadence/aot:ops_registrations",
            "fbcode//executorch/backends/cadence/aot:export_example",
            "fbcode//executorch/backends/cadence/aot:compiler",
            "fbcode//executorch/examples/cadence/operators:facto_util",
        ],
    )
