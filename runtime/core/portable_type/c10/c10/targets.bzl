load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime", "is_arvr_mode")

def get_sleef_preprocessor_flags():
    if runtime.is_oss:
        return []
    return ["-DAT_BUILD_ARM_VEC256_WITH_SLEEF"]


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
        visibility = ["//executorch/kernels/optimized/..."],
        exported_deps = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": [
                "fbsource//third-party/sleef:sleef_arm",
            ] if not runtime.is_oss else [],
            # fbsource//third-party/sleef:sleef currently fails to
            # link with missing symbols, hence the fbcode-specific dep below.
        }),
        fbcode_exported_deps = ([
            "//caffe2:aten-headers-cpu",
            "//caffe2:generated-config-header",
            "//caffe2/c10:c10_headers",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_64": [
                "third-party//sleef:sleef",
            ]
        })) if not runtime.is_oss else [],
        fbcode_exported_preprocessor_flags = [
            # We don't -DCPU_CAPABILITY=AVX2 because that trips
            # -Wmacro-redefined, and we only care about getting
            # reasonable vectorization and Sleef support.
            "-DCPU_CAPABILITY_AVX2",
            "-DET_USE_PYTORCH_HEADERS",
            "-DHAVE_AVX2_CPU_DEFINITION",
            "-DSTANDALONE_TORCH_HEADER",
        ] + get_sleef_preprocessor_flags(),
        xplat_exported_deps = [
            "//xplat/caffe2:aten_header",
            "//xplat/caffe2/c10:c10_headers",
        ] + ["//xplat/caffe2:ovrsource_aten_Config.h" if is_arvr_mode() else "//xplat/caffe2:generated_aten_config_header",],
        exported_preprocessor_flags = select({
            # Intentionally punting on non-fbcode x86 sleef support
            # for now because of fbsource//third-party/sleef:sleef
            # linker failure.
            "ovr_config//cpu:arm64": get_sleef_preprocessor_flags(),
            "DEFAULT": [],
        }) + ["-DSTANDALONE_TORCH_HEADER"] + ([] if runtime.is_oss else ["-DET_USE_PYTORCH_HEADERS"]),
    )
