load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Utility functions that can be used by operators that perform reduction
    runtime.cxx_library(
        name = "copy_ops_util",
        srcs = ["copy_ops_util.cpp"],
        exported_headers = [
            "copy_ops_util.h",
        ],
        compiler_flags = select({
                "DEFAULT": select({
                    "DEFAULT": ["-Wno-missing-prototypes"],
                    # GCC's C++ frontend rejects this C-only flag under -Werror.
                    # Nested under DEFAULT so windows and gcc can't both match.
                    "ovr_config//compiler:gcc": [],
                }),
                "ovr_config//os:windows": [],
            }) if not runtime.is_oss else select({
                "DEFAULT": ["-Wno-missing-prototypes"],
                "ovr_config//os:windows": [],
            }),
        deps = [
            "//executorch/runtime/kernel:kernel_includes_aten",
            "//executorch/runtime/core/exec_aten/util:tensor_util_aten",
        ],
        exported_preprocessor_flags = ["-DUSE_ATEN_LIB"],
        visibility = [
            "//executorch/kernels/aten/cpu/...",
            "//executorch/kernels/portable/cpu/...",
            "//executorch/kernels/optimized/cpu/...",
        ],
    )
