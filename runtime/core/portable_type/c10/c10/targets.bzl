load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime", "is_arvr_mode")

def get_preprocessor_flags(is_fbcode):
    flags = ["-DSTANDALONE_TORCH_HEADER"]
    if runtime.is_oss:
        return flags
    arm64_flags = [
        "-DCPU_CAPABILITY_DEFAULT",
    ]
    if is_fbcode:
        # TODO: enable Sleef in xplat?
        arm64_flags = arm64_flags + ["-DAT_BUILD_ARM_VEC256_WITH_SLEEF"]

    x86_avx2_flags = [
        "-DCPU_CAPABILITY_AVX2",
        "-DHAVE_AVX2_CPU_DEFINITION",
    ]
    default_flags = [
        "-DCPU_CAPABILITY_DEFAULT",
    ]
    fbcode_flags = select({
        "ovr_config//cpu:x86_64": x86_avx2_flags,
        "ovr_config//cpu:arm64": arm64_flags,
        "DEFAULT": default_flags,
    })
    non_fbcode_flags = select({
        "ovr_config//cpu/x86:avx2": x86_avx2_flags,
        "ovr_config//cpu:arm64": arm64_flags,
        "DEFAULT": default_flags,
    })
    return flags + ["-DET_USE_PYTORCH_HEADERS"] + (fbcode_flags if is_fbcode else non_fbcode_flags)

def get_sleef_deps():
    if runtime.is_oss:
        return []
    return select({
        "DEFAULT": [],
        "ovr_config//cpu:x86_64": [
            "fbsource//third-party/sleef:sleef",
        ],
        "ovr_config//cpu:arm64": [
            "fbsource//third-party/sleef:sleef",
        ],
    })

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_library(
        name = "c10",
        header_namespace = "c10",
        exported_headers = [
            "macros/Export.h",
            "macros/Macros.h",
            "util/BFloat16.h",
            "util/BFloat16-inl.h",
            "util/BFloat16-math.h",
            "util/Half.h",
            "util/Half-inl.h",
            "util/TypeSafeSignMath.h",
            "util/bit_cast.h",
            "util/complex.h",
            "util/complex_math.h",
            "util/complex_utils.h",
            "util/floating_point_utils.h",
            "util/irange.h",
        ],
        exported_preprocessor_flags = [
            "-DC10_USING_CUSTOM_GENERATED_MACROS",
        ] + ([] if runtime.is_oss else [
            "-DC10_USE_GLOG",
            "-DC10_USE_MINIMAL_GLOG",
        ]),
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = select({
            "DEFAULT": [],
            # Half-inl.h depends on vec_half.h from ATen, but only when building for x86.
            "ovr_config//cpu:x86_64": [
                ":aten_headers_for_executorch",
            ],
        }),
    )

    runtime.cxx_library(
        name = "aten_headers_for_executorch",
        srcs = [],
        visibility = ["//executorch/kernels/optimized/...", "@EXECUTORCH_CLIENTS"],
        # select() on ovr_config//runtime:fbcode does not work
        # properly in all cases. I have seen
        # //xplat/executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch
        # pass such a select in (at least) arvr mode. Going back to
        # fbcode_exported_deps accordingly.
        exported_deps = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": [
                "fbsource//third-party/sleef:sleef",
            ] if not runtime.is_oss else [],
        }),
        xplat_exported_deps = [
            "//xplat/caffe2:aten_header",
            "//xplat/caffe2/c10:c10_headers",
            ("//xplat/caffe2:ovrsource_aten_Config.h"
            if is_arvr_mode() else "//xplat/caffe2:generated_aten_config_header"),
        ], # + get_sleef_deps(), # TODO: enable Sleef in xplat?
        fbcode_exported_deps = ([
            "//caffe2:aten-headers-cpu",
            "//caffe2:generated-config-header",
            "//caffe2/c10:c10_headers",
        ] + get_sleef_deps()) if not runtime.is_oss else [],
        exported_preprocessor_flags = get_preprocessor_flags(is_fbcode=False)
        + ([] if runtime.is_oss else ["-DET_USE_PYTORCH_HEADERS"]),
        fbcode_exported_preprocessor_flags = get_preprocessor_flags(is_fbcode=True)
        + ([] if runtime.is_oss else ["-DET_USE_PYTORCH_HEADERS"]),
    )
