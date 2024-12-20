load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

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
            "util/floating_point_utils.h",
        ],
        exported_preprocessor_flags = [
            # NOTE: If we define C10_EMBEDDED to prevent Half and
            # BFloat16 from supporting streams, non-ExecuTorch-core
            # uses of other ATen headers that try to print ATen
            # primitive types fail to build because, apparently, there
            # are implicit conversions from Half/BFloat16 to a variety
            # of primitive types, not just float. Since merely
            # including <ostream> shouldn't result in any runtime
            # artifacts if stream code is never actually called, it
            # seems best to just not define C10_EMBEDDED, but if you
            # need it, it's there.
            # "-DC10_EMBEDDED",
            "-DC10_USE_GLOG",
            "-DC10_USE_MINIMAL_GLOG",
            "-DC10_USING_CUSTOM_GENERATED_MACROS",
        ],
        visibility = [
            "//executorch/runtime/core/portable_type/...",
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
        fbcode_exported_deps = [
            "//caffe2:aten-headers-cpu",
            "//caffe2:generated-config-header",
            "//caffe2/c10/core:base",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_64": [
                "third-party//sleef:sleef",
            ]
        }),
        xplat_exported_deps = [
            "//xplat/caffe2:aten_header",
            "//xplat/caffe2:generated_aten_config_header",
            "//xplat/caffe2/c10:c10",
        ],
        exported_preprocessor_flags = select({
            "ovr_config//cpu:x86_64": [
                "-DCPU_CAPABILITY=AVX2",
                "-DCPU_CAPABILITY_AVX2",
                "-DHAVE_AVX2_CPU_DEFINITION",
            ] + get_sleef_preprocessor_flags(),
            "ovr_config//cpu:arm64": get_sleef_preprocessor_flags(),
            "DEFAULT": [],
        }) + ["-DSTANDALONE_TORCH_HEADER"],
    )
