
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_library(
        name = "error_matchers",
        exported_headers = [
            "error_matchers.h",
        ],
        visibility = [
            "//executorch/runtime/core/testing_util/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        compiler_flags = [
            "-Wno-unneeded-internal-declaration",
        ],
        exported_external_deps = [
            "gmock",
        ],
    )
