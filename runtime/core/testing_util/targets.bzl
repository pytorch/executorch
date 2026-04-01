
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_library(
        name = "error_matchers",
        srcs = [
            "error_matchers.cpp",
        ],
        exported_headers = [
            "error_matchers.h",
        ],
        visibility = ["PUBLIC"],
        exported_external_deps = [
            "gmock",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ]
    )
