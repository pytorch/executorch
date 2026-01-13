# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//arvr/tools/build_defs:oxx.bzl", "oxx_binary", "oxx_static_library")


def define_common_targets():
    runtime.cxx_library(
        name = "vision-nnlib",
        srcs = select({
            "DEFAULT": ["dummy.c"],  # Use dummy file for non-Xtensa builds
            "ovr_config//cpu:xtensa": glob(["library/**/*.c"]),
        }),
        exported_headers = glob([
            "include/*.h", 
            "include_private/*.h"
        ]),
        header_namespace = "",
        visibility = ["PUBLIC"],
        platforms = CXX,
        compatible_with = select({
            "DEFAULT": [],
            "ovr_config//cpu:xtensa": ["ovr_config//cpu:xtensa"],
        }),
        compiler_flags = select({
            "DEFAULT": ["-UCOMPILER_XTENSA"],  # Ensure COMPILER_XTENSA is not defined for non-Xtensa builds
            "ovr_config//cpu:xtensa": [
                "-DCOMPILER_XTENSA",
                "-Ixplat/executorch/backends/cadence/vision/third-party/include",
                "-Ixplat/executorch/backends/cadence/vision/third-party/include_private",
            ],
        }),
        define_static_target = True,
    )
