load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.

    See README.md for instructions on selective build.
    """
    runtime.filegroup(
        name = "templates",
        srcs = native.glob([
            "templates/**/*.cpp",
            "templates/**/*.ini",
            "templates/**/*.h",
        ]),
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "macros",
        exported_headers = [
            "macros.h",
        ],
        visibility = [
            "//executorch/runtime/kernel/...",
            "//executorch/kernels/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
