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

def get_exported_linker_flags():
    if not runtime.is_oss:
        exported_linker_flags = select({
            "DEFAULT": [],
            "ovr_config//os:macos-arm64": ["-framework", "Accelerate"],
            "ovr_config//os:macos-x86_64": ["-framework", "Accelerate"],
        })
        return exported_linker_flags
    return []

def get_vec_preprocessor_flags():
    if not runtime.is_oss:
        # various ovr_configs are not available in oss
        preprocessor_flags = select({
            "ovr_config//os:linux-x86_64": [
                "-DET_BUILD_ARM_VEC256_WITH_SLEEF",
            ] if not runtime.is_oss else [],
            "ovr_config//os:iphoneos-arm64": [
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
            "ovr_config//os:iphoneos-arm64": [
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
    preprocessor_flags = select({
        "DEFAULT": [],
        "ovr_config//os:linux": [
            "-DCPU_CAPABILITY_AVX2",
        ],
    })
    return preprocessor_flags

def get_vec_fbcode_preprocessor_flags():
    preprocessor_flags = [
        "-DCPU_CAPABILITY_AVX2",
    ]
    return preprocessor_flags

def get_apple_framework_deps_kwargs(is_fbcode):
    # various ovr_configs are not available in oss
    if not runtime.is_oss and not is_fbcode:
        # Jump through few hoops since 'frameworks' is not a valid kwarg
        # for some buck rules
        frameworks = {'frameworks': select({
            "DEFAULT": [],
            "ovr_config//os:iphoneos": ["$SDKROOT/System/Library/Frameworks/Accelerate.framework"],
            "ovr_config//os:macos-arm64": ["$SDKROOT/System/Library/Frameworks/Accelerate.framework"],
            "ovr_config//os:macos-x86_64": ["$SDKROOT/System/Library/Frameworks/Accelerate.framework"],
        })}
        return frameworks
    return {'fbobjc_frameworks': ["Accelerate"]}

def get_preprocessor_flags():
    # various ovr_configs are not available in oss
    preprocessor_flags = select({
      ":linux-x86_64": [
          "-DET_BUILD_WITH_BLAS",
      ] if not runtime.is_oss else [],
      "DEFAULT": [],
    })

    if not runtime.is_oss:
        # various ovr_configs are not available in oss
        additional_preprocessor_flags = select({
            "ovr_config//os:iphoneos": [
                "-DET_BUILD_WITH_BLAS",
                "-DET_BUILD_FOR_APPLE",
            ] if not runtime.is_oss else [],
            "ovr_config//os:macos-arm64": [
                "-DET_BUILD_WITH_BLAS",
                "-DET_BUILD_FOR_APPLE",
            ] if not runtime.is_oss else [],
            "ovr_config//os:macos-x86_64": [
                "-DET_BUILD_WITH_BLAS",
                "-DET_BUILD_FOR_APPLE",
            ] if not runtime.is_oss else [],
            "DEFAULT": [],
        })
        preprocessor_flags = preprocessor_flags + additional_preprocessor_flags
    return preprocessor_flags


# TODO(ssjia): Enable -DCPU_CAPABILITY_AVX2 in fbcode, which requires sleef.
def define_libs(is_fbcode=False):
    runtime.cxx_library(
        name = "libvec",
        srcs = [],
        exported_headers = native.glob([
            "vec/**/*.h",
        ]),
        header_namespace = "executorch/kernels/optimized",
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
        ],
        deps = select({
            "DEFAULT": [],
            "ovr_config//os:linux": [
                "fbsource//third-party/sleef:sleef",
            ] if not runtime.is_oss else [],
        }) + select({
            "DEFAULT": [],
            "ovr_config//os:android": select({
                "DEFAULT": [],
                "ovr_config//cpu:arm64": [
                    "fbsource//third-party/sleef:sleef",
                ] if not runtime.is_oss else [],
            }),
        }),
    )

    runtime.cxx_library(
        name = "libutils",
        srcs = [],
        exported_headers = native.glob([
            "utils/**/*.h",
        ]),
        header_namespace = "executorch/kernels/optimized",
        visibility = ["PUBLIC"],
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

    LIBBLAS_DEPS = [
        third_party_dep("cpuinfo"),
        "//executorch/extension/threadpool:threadpool",
    ]

    for libblas_name, mkl_dep in [("libblas", "fbsource//third-party/mkl:mkl_lp64_omp"), ("libblas_mkl_noomp", "fbsource//third-party/mkl:mkl")]:
        # Merge platform-specific kwargs
        platform_kwargs = get_apple_framework_deps_kwargs(is_fbcode)
        if not is_fbcode:
            platform_kwargs.update({
                "fbandroid_preprocessor_flags": select({
                    "DEFAULT": [],
                    "ovr_config//os:android": select({
                        "DEFAULT": [],
                        "ovr_config//cpu:arm64": [
                            "-DET_BUILD_WITH_BLAS",
                        ],
                    }),
                }),
                "fbandroid_deps": select({
                    "DEFAULT": [],
                    "ovr_config//os:android": select({
                        "DEFAULT": [],
                        "ovr_config//cpu:arm64": [
                            "fbsource//arvr/third-party/eigen:eigen3_blas",
                        ],
                    }),
                }),
            })

        runtime.cxx_library(
            name = libblas_name,
            srcs = native.glob([
                "blas/**/*.cpp",
            ]),
            exported_headers = native.glob([
                "blas/**/*.h",
            ]),
            exported_linker_flags = get_exported_linker_flags(),
            compiler_flags = ["-Wno-pass-failed"] + select({
                "ovr_config//runtime:fbcode": [],
                # TODO: replace with get_compiler_optimization_flags from op_registration_util.bzl when that
                # is re-enabled.
                "DEFAULT": ["-Os"],
            }),
            header_namespace = "executorch/kernels/optimized",
            visibility = ["PUBLIC"],
            preprocessor_flags = get_preprocessor_flags(),
            fbobjc_exported_preprocessor_flags = [
                "-DET_BUILD_WITH_BLAS",
                "-DET_BUILD_FOR_APPLE",
            ],
            deps = select({
                ":linux-x86_64": [mkl_dep] if not runtime.is_oss else [],
                "DEFAULT": [],
            }) + LIBBLAS_DEPS,
            exported_deps = [
                "//executorch/extension/threadpool:threadpool",
                "//executorch/kernels/optimized:libutils",
                "//executorch/runtime/core/exec_aten:lib",
                "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
            ],
            **platform_kwargs,
        )
