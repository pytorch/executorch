# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under both the MIT license found in the
# LICENSE-MIT file in the root directory of this source tree and the Apache
# License, Version 2.0 found in the LICENSE-APACHE file in the root directory
# of this source tree.

load("@fbsource//tools/build_defs:default_platform_defs.bzl", "DEVSERVER_PLATFORM_REGEX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Because vec exists as a collection of header files, compile and preprocessor
# flags applied to the vec target do not have any effect, since no compilation
# actually occurs for the target.
#
# Targets using the vec library must therefore call the get_vec_*_flags
# functions in order to declare the required compiler flags needed in order to
# access CPU vector intrinsics.

# This oopy from kernels/optimized/lib_defs.bzl is not necessary.
# This file really needs to be removed
def get_vec_preprocessor_flags():
    if not runtime.is_oss:
        # various ovr_configs are not available in oss
        preprocessor_flags = select({
            "ovr_config//os:iphoneos": [
                "-DET_BUILD_ARM_VEC256_WITH_SLEEF",
            ] if not runtime.is_oss else [],
            "ovr_config//os:macos-arm64": [
                "-DET_BUILD_ARM_VEC256_WITH_SLEEF",
            ] if not runtime.is_oss else [],
            "ovr_config//os:android-arm64": [
                "-DET_BUILD_ARM_VEC256_WITH_SLEEF",
            ] if not runtime.is_oss else [],
            "DEFAULT": [],
        })
        return preprocessor_flags
    return []

def get_vec_deps():
    if not runtime.is_oss:
        # various ovr_configs are not available in oss
        deps = select({
            "ovr_config//os:linux-x86_64": [
                "fbsource//third-party/sleef:sleef",
            ] if not runtime.is_oss else [],
            "ovr_config//os:iphoneos": [
                "fbsource//third-party/sleef:sleef",
            ] if not runtime.is_oss else [],
            "ovr_config//os:macos-arm64": [
                "fbsource//third-party/sleef:sleef",
            ] if not runtime.is_oss else [],
            "ovr_config//os:android-arm64": [
                "fbsource//third-party/sleef:sleef",
            ] if not runtime.is_oss else [],
            "DEFAULT": [],
        })
        return deps
    return []

def get_vec_cxx_preprocessor_flags():
    preprocessor_flags = [
        (
            DEVSERVER_PLATFORM_REGEX,
            [
                "-DCPU_CAPABILITY_AVX2",
            ],
        ),
    ]
    return preprocessor_flags

def get_vec_fbcode_preprocessor_flags():
    preprocessor_flags = select({
        "ovr_config//cpu/x86:avx2": [
            "-DCPU_CAPABILITY_AVX2",
        ],
        "DEFAULT": [],
    })
    return preprocessor_flags

# TODO(ssjia): Enable -DCPU_CAPABILITY_AVX2 in fbcode, which requires sleef.
def define_libs():
    runtime.cxx_library(
        name = "libvec",
        srcs = [],
        exported_headers = native.glob([
            "vec/**/*.h",
        ]),
        header_namespace = "executorch/kernels/optimized",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        cxx_platform_deps = select({
            "DEFAULT": [
                (
                    DEVSERVER_PLATFORM_REGEX,
                    [
                        "fbsource//third-party/sleef:sleef",
                    ],
                ),
            ],
            "ovr_config//cpu:arm64": [
                (
                    DEVSERVER_PLATFORM_REGEX,
                    [
                        "fbsource//third-party/sleef:sleef",
                    ],
                ),
            ],
        }),
        fbandroid_platform_deps = [
            (
                "^android-arm64.*$",
                [
                    "fbsource//third-party/sleef:sleef",
                ],
            ),
        ],
    )

    runtime.cxx_library(
        name = "libutils",
        srcs = [],
        exported_headers = native.glob([
            "utils/**/*.h",
        ]),
        header_namespace = "executorch/kernels/optimized",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            # Needed to access the ET_INLINE macro
            "//executorch/runtime/platform:compiler",
        ],
    )

    runtime.cxx_library(
        name = "libblas",
        srcs = native.glob([
            "blas/**/*.cpp",
        ]),
        exported_headers = native.glob([
            "blas/**/*.h",
        ]),
        header_namespace = "executorch/kernels/optimized",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        fbandroid_platform_preprocessor_flags = [
            (
                "^android-arm64.*$",
                [
                    "-DET_BUILD_WITH_BLAS",
                ],
            ),
        ],
        fbandroid_platform_deps = [
            (
                "^android-arm64.*$",
                [
                    "fbsource//third-party/openblas:openblas",
                ],
            ),
        ],
        fbobjc_exported_preprocessor_flags = [
            "-DET_BUILD_WITH_BLAS",
            "-DET_BUILD_FOR_APPLE",
        ],
        fbobjc_frameworks = [
            "Accelerate",
        ],
        exported_deps = [
            "//executorch/kernels/optimized:libutils",
            "//executorch/runtime/core/exec_aten:lib",
        ],
    )
