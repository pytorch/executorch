# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//arvr/tools/build_defs:oxx.bzl", "oxx_binary", "oxx_static_library")


def define_common_targets():
    oxx_static_library(
        name = "vision-nnlib",
        srcs = glob([
            "library/**/*.c"
        ]),
        public_include_directories = [
            "include",
            "include_private"
        ],
        public_raw_headers = glob([
            "include/*.h", 
            "include_private/*.h"
        ]),
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
    )
