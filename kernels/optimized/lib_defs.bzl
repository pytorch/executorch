load("@fbsource//tools/build_defs:default_platform_defs.bzl", "DEVSERVER_PLATFORM_REGEX")
load("@fbsource//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Because vec exists as a collection of header files, compile and preprocessor
# flags applied to the vec target do not have any effect, since no compilation
# actually occurs for the target.
#
# Targets using the vec library must therefore call the get_vec_*_flags
# functions in order to declare the required compiler flags needed in order to
# access CPU vector intrinsics.

def get_vec_android_preprocessor_flags():
    preprocessor_flags = [
        (
            "^android-arm64.*$",
            [
                "-DET_BUILD_ARM_VEC256_WITH_SLEEF",
            ],
        ),
    ]
    return preprocessor_flags

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
    preprocessor_flags = [
        "-DCPU_CAPABILITY_AVX2",
    ]
    return preprocessor_flags

# Currently, having a dependency on fbsource//third-party/sleef:sleef may cause
# duplicate symbol errors when linking fbcode targets in opt mode that also
# depend on ATen. This is because ATen accesses sleef via the third-party folder
# in caffe2 (caffe2/third-party//sleef:sleef).
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
                        "fbsource//third-party/sleef:sleef_arm",
                    ],
                ),
            ],
        }),
        fbandroid_platform_deps = [
            (
                "^android-arm64.*$",
                [
                    "fbsource//third-party/sleef:sleef_arm",
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

    # OSS doesn't have ovr_config//os:linux-x86_64
    fb_native.config_setting(
        name = "linux-x86_64",
        constraint_values = [
            "ovr_config//os/constraints:linux",
            "ovr_config//cpu/constraints:x86_64",
        ],
    )

    LIBBLAS_DEPS = [third_party_dep("cpuinfo")]

    for libblas_name, mkl_dep in [("libblas", "fbsource//third-party/mkl:mkl_lp64_omp"), ("libblas_mkl_noomp", "fbsource//third-party/mkl:mkl")]:
        runtime.cxx_library(
            name = libblas_name,
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
            preprocessor_flags = select({
                ":linux-x86_64": [
                    "-DET_BUILD_WITH_BLAS",
                ] if not runtime.is_oss else [],
                "DEFAULT": [],
            }),
            fbandroid_platform_compiler_flags = [
                (
                    "^android-arm64.*$",
                    [
                        "-march=armv8+bf16",
                    ],
                ),
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
            fbobjc_compiler_flags = [
                "-march=armv8+bf16",
            ],
            fbobjc_exported_preprocessor_flags = [
                "-DET_BUILD_WITH_BLAS",
                "-DET_BUILD_FOR_APPLE",
            ],
            fbobjc_frameworks = [
                "Accelerate",
            ],
            deps = select({
                ":linux-x86_64": [mkl_dep] if not runtime.is_oss else [],
                "DEFAULT": [],
            }) + LIBBLAS_DEPS,
            exported_deps = [
                "//executorch/extension/parallel:thread_parallel",
                "//executorch/kernels/optimized:libutils",
                "//executorch/runtime/core/exec_aten:lib",
            ],
        )
