load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

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
    )
