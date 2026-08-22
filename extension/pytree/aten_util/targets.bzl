load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "ivalue_util",
        srcs = ["ivalue_util.cpp"],
        exported_headers = ["ivalue_util.h"],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/extension/pytree:pytree",
            "//executorch/runtime/platform:platform",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        external_deps = [
            "torch-core-cpp",
        ],
    )
